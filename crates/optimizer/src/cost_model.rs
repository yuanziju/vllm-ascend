//! cost_model — 算子开销估算模型

use base::{Graph, NodeView, OpKind, Result, ValueId};

#[derive(Debug, Clone, Copy)]
pub struct CostCoeffs {
    pub flops: f64,
    pub mem: f64,
    pub launch: f64,
}

impl CostCoeffs {
    pub fn cuda() -> Self {
        Self {
            flops: 1.0,
            mem: 2.5,
            launch: 10.0,
        }
    }
    pub fn npu() -> Self {
        Self {
            flops: 0.8,
            mem: 2.0,
            launch: 5.0,
        }
    }
    pub fn cpu() -> Self {
        Self {
            flops: 1.0,
            mem: 1.0,
            launch: 0.0,
        }
    }
    pub fn for_target(target: common::Target) -> Self {
        match target {
            common::Target::Cuda => Self::cuda(),
            common::Target::Npu => Self::npu(),
            common::Target::Cpu => Self::cpu(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OpCost {
    pub flops: f64,
    pub mem_bytes: f64,
    pub launch: f64,
}

impl OpCost {
    pub fn total(&self, c: CostCoeffs) -> f64 {
        c.flops * self.flops + c.mem * self.mem_bytes + c.launch * self.launch
    }
}

pub fn estimate_op(graph: &Graph, node: NodeView) -> Result<OpCost> {
    let mut in_bytes = 0.0f64;
    for &vin in node.inputs() {
        if let Ok(v) = graph.value(vin) {
            in_bytes += value_bytes(v) as f64;
        }
    }
    let mut out_bytes = 0.0f64;
    for &vout in node.outputs() {
        if let Ok(v) = graph.value(vout) {
            out_bytes += value_bytes(v) as f64;
        }
    }
    let mem = in_bytes + out_bytes;

    let (flops, launch) = match node.kind {
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => (out_bytes / 4.0, 1.0),
        OpKind::Relu => (out_bytes / 4.0, 1.0),
        OpKind::Gelu => (out_bytes / 4.0 * 8.0, 1.0),
        OpKind::Sigmoid => (out_bytes / 4.0 * 4.0, 1.0),
        OpKind::Tanh => (out_bytes / 4.0 * 6.0, 1.0),
        // Rsqrt：1/sqrt(x) 单 op，常硬件单指令或 0x5f3759df 位 trick，比 Sqrt+Div 便宜
        OpKind::Rsqrt => (out_bytes / 4.0 * 2.0, 1.0),
        OpKind::Softmax => (out_bytes / 4.0 * 12.0, 1.0),
        OpKind::MatMul => {
            // [m,k] × [k,n] → [m,n]，FLOPs = 2·m·n·k（需双输入 shape 已知）
            // shape 未知时退化到方阵估计（out 是 n×n，FLOPs ≈ 2n³）
            let flops = matmul_flops(graph, node.inputs()).unwrap_or_else(|| {
                let s = (out_bytes / 4.0).sqrt();
                s * s * s * 2.0
            });
            (flops, 1.0)
        }
        OpKind::LayerNorm => (out_bytes / 4.0 * 16.0, 1.0),
        OpKind::Conv => (out_bytes / 4.0 * 100.0, 1.0),
        OpKind::Reshape | OpKind::Transpose | OpKind::Concat | OpKind::Slice => (0.0, 1.0),
        // Reduce 类：读全部输入元素，flops 与输入量成正比（sum/max 简单累加）
        OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax => (in_bytes / 4.0 * 2.0, 1.0),
        OpKind::Constant | OpKind::Placeholder | OpKind::Return => (0.0, 0.0),
        _ => (out_bytes / 4.0, 1.0),
    };

    Ok(OpCost {
        flops,
        mem_bytes: mem,
        launch,
    })
}

/// MatMul FLOPs = 2·m·n·k（[m,k]×[k,n]→[m,n]）。返回 None 表示输入 shape 未知
/// 或输入不足，调用方应退化到方阵估计。
fn matmul_flops(graph: &Graph, ins: &[ValueId]) -> Option<f64> {
    let a = graph.value(*ins.first()?).ok()?;
    let b = graph.value(*ins.get(1)?).ok()?;
    let sa = a.shape();
    let sb = b.shape();
    if sa.len() < 2 || sb.len() < 2 {
        return None;
    }
    let m = sa[sa.len() - 2];
    let k = sa[sa.len() - 1];
    let n = sb[sb.len() - 1];
    if m <= 0 || k <= 0 || n <= 0 {
        return None;
    }
    Some(2.0 * (m as f64) * (n as f64) * (k as f64))
}

fn value_bytes(v: base::ValueView) -> usize {
    let elem_bytes = match v.dtype() {
        base::DType::F32 | base::DType::I32 => 4,
        base::DType::F16 | base::DType::BF16 => 2,
        base::DType::I64 => 8,
        base::DType::Bool => 1,
    };
    let elems: usize = if v.is_tensor() {
        let shape = v.shape();
        if shape.is_empty() || shape.iter().any(|&d| d < 0) {
            1
        } else {
            shape.iter().map(|&d| d as usize).product()
        }
    } else {
        1
    };
    elem_bytes * elems
}

pub fn estimate_graph(graph: &Graph, coeffs: CostCoeffs) -> Result<f64> {
    let mut total = 0.0;
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        let c = estimate_op(graph, n)?;
        total += c.total(coeffs);
    }
    Ok(total)
}

pub fn fusion_saving(graph: &Graph, nodes: &[base::NodeId], coeffs: CostCoeffs) -> Result<f64> {
    if nodes.len() < 2 {
        return Ok(0.0);
    }
    let saved_launch = (nodes.len() - 1) as f64 * coeffs.launch;
    let mut mid_bytes = 0.0;
    for &id in &nodes[..nodes.len() - 1] {
        let n = graph.node(id)?;
        for &vout in n.outputs() {
            if let Ok(v) = graph.value(vout) {
                mid_bytes += value_bytes(v) as f64;
            }
        }
    }
    let saved_mem = mid_bytes * 2.0 * coeffs.mem;
    Ok(saved_launch + saved_mem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    fn tensor(dims: Vec<i64>) -> Type {
        Type::Tensor {
            dtype: DType::F32,
            dims,
        }
    }

    #[test]
    fn cuda_has_higher_launch() {
        assert!(CostCoeffs::cuda().launch > CostCoeffs::cpu().launch);
    }

    #[test]
    fn matmul_flops_uses_input_shapes() {
        // [2,3] × [3,4] → [2,4]，FLOPs = 2·m·n·k = 2·2·4·3 = 48
        let mut g = Graph::new("test");
        let a = g.add_input(tensor(vec![2, 3]), Some("a"));
        let b = g.add_input(tensor(vec![3, 4]), Some("b"));
        let mm = g.add_node(OpKind::MatMul);
        let out = g.add_value(tensor(vec![2, 4]), Some("o"), mm);
        g.storage.set_node_inputs(mm, &[a, b]);
        g.storage.set_node_outputs(mm, &[out]);
        let cost = estimate_op(&g, g.node(mm).unwrap()).unwrap();
        assert_eq!(cost.flops, 48.0, "MatMul [2,3]×[3,4] FLOPs 应为 2·2·4·3=48");
    }

    #[test]
    fn matmul_flops_falls_back_when_shape_unknown() {
        // 输入 shape 未知（含 -1），应退化到方阵估计，不报错
        let mut g = Graph::new("test");
        let a = g.add_input(tensor(vec![-1, -1]), Some("a"));
        let b = g.add_input(tensor(vec![-1, -1]), Some("b"));
        let mm = g.add_node(OpKind::MatMul);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), mm);
        g.storage.set_node_inputs(mm, &[a, b]);
        g.storage.set_node_outputs(mm, &[out]);
        // 不应 panic，应给出某个有限值（方阵退化）
        let cost = estimate_op(&g, g.node(mm).unwrap()).unwrap();
        assert!(cost.flops.is_finite(), "shape 未知时应退化给出有限值");
    }

    #[test]
    fn matmul_flops_non_square() {
        // 非方阵 [8,4] × [4,16] → [8,16]，FLOPs = 2·8·16·4 = 1024
        // 旧方阵估计会用 out_bytes/4 = 8*16 = 128 → sqrt=11.3 → 2*11.3³≈2896（错）
        let mut g = Graph::new("test");
        let a = g.add_input(tensor(vec![8, 4]), Some("a"));
        let b = g.add_input(tensor(vec![4, 16]), Some("b"));
        let mm = g.add_node(OpKind::MatMul);
        let out = g.add_value(tensor(vec![8, 16]), Some("o"), mm);
        g.storage.set_node_inputs(mm, &[a, b]);
        g.storage.set_node_outputs(mm, &[out]);
        let cost = estimate_op(&g, g.node(mm).unwrap()).unwrap();
        assert_eq!(cost.flops, 1024.0, "非方阵 MatMul 应用 m·n·k 而非方阵估计");
    }
}
