//! cost_model — 算子开销估算模型

use base::{Graph, NodeView, OpKind, Result};

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
        OpKind::Softmax => (out_bytes / 4.0 * 12.0, 1.0),
        OpKind::MatMul => {
            let n = (out_bytes / 4.0).sqrt();
            (n * n * n * 2.0, 1.0)
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

    #[test]
    fn cuda_has_higher_launch() {
        assert!(CostCoeffs::cuda().launch > CostCoeffs::cpu().launch);
    }
}
