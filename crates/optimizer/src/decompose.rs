//! decompose — 一对多拆分（LayerNorm/Softmax/Gelu → 细粒度原语）
//!
//! 设计哲学：decompose 阶段把高层复合算子拆成基础原语（Add/Sub/Mul/Div/
//! Sqrt/Exp/ReduceSum/ReduceMean/ReduceMax 等），让后续 algebra/CSE/fuse 能
//! 在细粒度上做通用优化。这是"一对多拆细"，不是贪心模式匹配。
//!
//! 拆分规则（数学等价）：
//! - **LayerNorm(x, γ, β, axis=-1, ε)**：
//!   mean = ReduceMean(x, axis); xc = x - mean; var = ReduceMean(xc*xc, axis);
//!   std = Sqrt(var + ε); inv = 1/std; norm = xc*inv; scaled = norm*γ; out = scaled+β
//! - **Softmax(x, axis)**（数值稳定版，用 max 技巧）：
//!   m = ReduceMax(x, axis); shifted = x - m; e = Exp(shifted);
//!   s = ReduceSum(e, axis); out = e / s
//! - **Gelu(x)**（tanh 近似，避免 erf）：
//!   c = sqrt(2/π) ≈ 0.7978845608; t = x + 0.044715*x³;
//!   out = 0.5 * x * (1 + Tanh(c * t))

use base::StorageAttrKey;
use base::{Graph, NodeId, NodeView, OpKind, Result, ValueId};
use std::collections::HashSet;

pub struct DecomposeResult {
    pub original: NodeId,
    pub expanded: Vec<NodeId>,
}

pub fn run_decompose(graph: &mut Graph) -> Result<Vec<DecomposeResult>> {
    let mut results = Vec::new();

    // 第一阶段：纯读取，收集待拆节点（避免 borrow 冲突）
    let to_decompose: Vec<(NodeId, OpKind)> = graph
        .node_ids()
        .filter_map(|id| {
            graph.node(id).ok().and_then(|n| {
                if matches!(n.kind, OpKind::LayerNorm | OpKind::Softmax | OpKind::Gelu) {
                    Some((id, n.kind))
                } else {
                    None
                }
            })
        })
        .collect();

    let mut to_remove: HashSet<NodeId> = HashSet::new();
    for (id, kind) in to_decompose {
        let expanded = match kind {
            OpKind::LayerNorm => decompose_layernorm(graph, id)?,
            OpKind::Softmax => decompose_softmax(graph, id)?,
            OpKind::Gelu => decompose_gelu(graph, id)?,
            _ => Vec::new(),
        };
        if !expanded.is_empty() {
            to_remove.insert(id);
            results.push(DecomposeResult {
                original: id,
                expanded,
            });
        }
    }

    // 物理删除原节点（已被细粒度子图替代）
    if !to_remove.is_empty() {
        let (new_graph, _, _) = graph.compact(&to_remove);
        *graph = new_graph;
    }
    Ok(results)
}

// --- 属性读取辅助 ---

/// 读取节点的 Axis 属性（Int），默认 -1（最后一轴）
fn read_axis(node: NodeView) -> i64 {
    for e in node.attrs() {
        if e.key == StorageAttrKey::Axis as u8 && e.tag == base::storage::AttrTag::Int as u8 {
            return node.storage.attr_int(e);
        }
    }
    -1
}

/// 读取节点的 Epsilon 属性（Float），默认 1e-5
fn read_epsilon(node: NodeView) -> f64 {
    for e in node.attrs() {
        if e.key == StorageAttrKey::Epsilon as u8 && e.tag == base::storage::AttrTag::Float as u8 {
            return node.storage.attr_float(e);
        }
    }
    1e-5
}

// --- 子图构建辅助 ---

/// 构造一个 reduce 节点（ReduceSum/ReduceMean/ReduceMax）：
/// 单输入 in_v，输出 value 类型沿用 out_type，带 Axis 属性
fn build_reduce(
    graph: &mut Graph,
    kind: OpKind,
    in_v: ValueId,
    out_type: base::Type,
    axis: i64,
    name: Option<&str>,
) -> Result<(NodeId, ValueId)> {
    let node = graph.add_node(kind);
    let out = graph.add_value(out_type, name, node);
    graph.storage.set_node_inputs(node, &[in_v]);
    graph.storage.set_node_outputs(node, &[out]);
    graph.storage.add_attr_int(node, StorageAttrKey::Axis, axis);
    Ok((node, out))
}

/// 构造二元 elementwise 节点（Add/Sub/Mul/Div）：输出类型取 a 的类型
fn build_binop(
    graph: &mut Graph,
    kind: OpKind,
    a: ValueId,
    b: ValueId,
    out_type: base::Type,
    name: Option<&str>,
) -> Result<(NodeId, ValueId)> {
    let node = graph.add_node(kind);
    let out = graph.add_value(out_type, name, node);
    graph.storage.set_node_inputs(node, &[a, b]);
    graph.storage.set_node_outputs(node, &[out]);
    Ok((node, out))
}

/// 构造单输入 elementwise 节点（Exp/Sqrt/Tanh 等）
fn build_unop(
    graph: &mut Graph,
    kind: OpKind,
    in_v: ValueId,
    out_type: base::Type,
    name: Option<&str>,
) -> Result<(NodeId, ValueId)> {
    let node = graph.add_node(kind);
    let out = graph.add_value(out_type, name, node);
    graph.storage.set_node_inputs(node, &[in_v]);
    graph.storage.set_node_outputs(node, &[out]);
    Ok((node, out))
}

// =========================================================================
// LayerNorm 拆分
// =========================================================================

fn decompose_layernorm(graph: &mut Graph, id: NodeId) -> Result<Vec<NodeId>> {
    // 先收集原节点信息（不可变借用），再构建子图（可变借用）
    let (inputs, out_type, axis, eps, original_outputs) = {
        let n = graph.node(id)?;
        let ins = n.inputs().to_vec();
        let outs = n.outputs().to_vec();
        // 输出类型沿用原节点第一个输出的类型
        let ot = if outs.is_empty() {
            // 退化：从输入推断
            base::Type::Scalar(base::DType::F32)
        } else {
            type_of_value(graph, outs[0])?
        };
        (ins, ot, read_axis(n), read_epsilon(n), outs)
    };

    // LayerNorm 期望 inputs = [x, gamma, beta]
    if inputs.len() < 3 {
        return Ok(Vec::new());
    }
    let (x, gamma, beta) = (inputs[0], inputs[1], inputs[2]);

    // mean = ReduceMean(x, axis)
    let (_mean_n, mean) = build_reduce(
        graph,
        OpKind::ReduceMean,
        x,
        out_type.clone(),
        axis,
        Some("ln_mean"),
    )?;
    // xc = x - mean
    let (_xc_n, xc) = build_binop(graph, OpKind::Sub, x, mean, out_type.clone(), Some("ln_xc"))?;
    // sq = xc * xc
    let (_sq_n, sq) = build_binop(graph, OpKind::Mul, xc, xc, out_type.clone(), Some("ln_sq"))?;
    // var = ReduceMean(sq, axis)
    let (_var_n, var) = build_reduce(
        graph,
        OpKind::ReduceMean,
        sq,
        out_type.clone(),
        axis,
        Some("ln_var"),
    )?;
    // var_eps = var + eps （常量）
    let (_eps_c, eps_v) = graph.add_constant_f64(eps);
    let (_ve_n, var_eps) = build_binop(
        graph,
        OpKind::Add,
        var,
        eps_v,
        out_type.clone(),
        Some("ln_var_eps"),
    )?;
    // std = Sqrt(var_eps)
    let (_std_n, std) = build_unop(
        graph,
        OpKind::Sqrt,
        var_eps,
        out_type.clone(),
        Some("ln_std"),
    )?;
    // inv = 1 / std （用 Constant 1 + Div，float_opts 后续可改 mul）
    let (_one_c, one_v) = graph.add_constant_f64(1.0);
    let (_inv_n, inv) = build_binop(
        graph,
        OpKind::Div,
        one_v,
        std,
        out_type.clone(),
        Some("ln_inv"),
    )?;
    // norm = xc * inv
    let (_norm_n, norm) = build_binop(
        graph,
        OpKind::Mul,
        xc,
        inv,
        out_type.clone(),
        Some("ln_norm"),
    )?;
    // scaled = norm * gamma
    let (_scaled_n, scaled) = build_binop(
        graph,
        OpKind::Mul,
        norm,
        gamma,
        out_type.clone(),
        Some("ln_scaled"),
    )?;
    // out = scaled + beta
    let (out_n, out_v) = build_binop(
        graph,
        OpKind::Add,
        scaled,
        beta,
        out_type.clone(),
        Some("ln_out"),
    )?;

    // 重写原节点 outputs 的所有使用者为新 out_v
    rewrite_value_uses(graph, &original_outputs, out_v);

    Ok(vec![
        _mean_n, _xc_n, _sq_n, _var_n, _ve_n, _std_n, _inv_n, _norm_n, _scaled_n, out_n,
    ])
}

// =========================================================================
// Softmax 拆分（数值稳定版）
// =========================================================================

fn decompose_softmax(graph: &mut Graph, id: NodeId) -> Result<Vec<NodeId>> {
    let (inputs, out_type, axis, original_outputs) = {
        let n = graph.node(id)?;
        let ins = n.inputs().to_vec();
        let outs = n.outputs().to_vec();
        let ot = if outs.is_empty() {
            base::Type::Scalar(base::DType::F32)
        } else {
            type_of_value(graph, outs[0])?
        };
        (ins, ot, read_axis(n), outs)
    };

    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let x = inputs[0];

    // m = ReduceMax(x, axis)
    let (_m_n, m) = build_reduce(
        graph,
        OpKind::ReduceMax,
        x,
        out_type.clone(),
        axis,
        Some("sm_max"),
    )?;
    // shifted = x - m
    let (_sh_n, shifted) = build_binop(
        graph,
        OpKind::Sub,
        x,
        m,
        out_type.clone(),
        Some("sm_shifted"),
    )?;
    // e = Exp(shifted)
    let (_e_n, e) = build_unop(graph, OpKind::Exp, shifted, out_type.clone(), Some("sm_e"))?;
    // s = ReduceSum(e, axis)
    let (_s_n, s) = build_reduce(
        graph,
        OpKind::ReduceSum,
        e,
        out_type.clone(),
        axis,
        Some("sm_sum"),
    )?;
    // out = e / s
    let (out_n, out_v) = build_binop(graph, OpKind::Div, e, s, out_type.clone(), Some("sm_out"))?;

    rewrite_value_uses(graph, &original_outputs, out_v);

    Ok(vec![_m_n, _sh_n, _e_n, _s_n, out_n])
}

// =========================================================================
// Gelu 拆分（tanh 近似）
// =========================================================================

/// tanh 近似常数
const GELU_C: f64 = 0.7978845608028654; // sqrt(2/π)
const GELU_K: f64 = 0.044715;

fn decompose_gelu(graph: &mut Graph, id: NodeId) -> Result<Vec<NodeId>> {
    let (inputs, out_type, original_outputs) = {
        let n = graph.node(id)?;
        let ins = n.inputs().to_vec();
        let outs = n.outputs().to_vec();
        let ot = if outs.is_empty() {
            base::Type::Scalar(base::DType::F32)
        } else {
            type_of_value(graph, outs[0])?
        };
        (ins, ot, outs)
    };

    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let x = inputs[0];

    // x³ = x * x * x
    let (_x2_n, x2) = build_binop(graph, OpKind::Mul, x, x, out_type.clone(), Some("gelu_x2"))?;
    let (_x3_n, x3) = build_binop(graph, OpKind::Mul, x2, x, out_type.clone(), Some("gelu_x3"))?;
    // k * x³
    let (_kc, k_v) = graph.add_constant_f64(GELU_K);
    let (_kx3_n, kx3) = build_binop(
        graph,
        OpKind::Mul,
        x3,
        k_v,
        out_type.clone(),
        Some("gelu_kx3"),
    )?;
    // t = x + k*x³
    let (_t_n, t) = build_binop(graph, OpKind::Add, x, kx3, out_type.clone(), Some("gelu_t"))?;
    // c * t
    let (_cc, c_v) = graph.add_constant_f64(GELU_C);
    let (_ct_n, ct) = build_binop(
        graph,
        OpKind::Mul,
        t,
        c_v,
        out_type.clone(),
        Some("gelu_ct"),
    )?;
    // tanh(c*t)
    let (_th_n, th) = build_unop(graph, OpKind::Tanh, ct, out_type.clone(), Some("gelu_tanh"))?;
    // 1 + tanh
    let (_one_c, one_v) = graph.add_constant_f64(1.0);
    let (_1p_n, one_plus) = build_binop(
        graph,
        OpKind::Add,
        one_v,
        th,
        out_type.clone(),
        Some("gelu_1pt"),
    )?;
    // 0.5 * x
    let (_half_c, half_v) = graph.add_constant_f64(0.5);
    let (_hx_n, hx) = build_binop(
        graph,
        OpKind::Mul,
        half_v,
        x,
        out_type.clone(),
        Some("gelu_halfx"),
    )?;
    // out = 0.5*x * (1+tanh)
    let (out_n, out_v) = build_binop(
        graph,
        OpKind::Mul,
        hx,
        one_plus,
        out_type.clone(),
        Some("gelu_out"),
    )?;

    rewrite_value_uses(graph, &original_outputs, out_v);

    Ok(vec![
        _x2_n, _x3_n, _kx3_n, _t_n, _ct_n, _th_n, _1p_n, _hx_n, out_n,
    ])
}

// =========================================================================
// 通用辅助
// =========================================================================

/// 读取一个 value 的类型（Type）
fn type_of_value(graph: &Graph, v: ValueId) -> Result<base::Type> {
    let val = graph.value(v)?;
    let dtype = val.dtype();
    if val.is_tensor() {
        Ok(base::Type::Tensor {
            dtype,
            dims: val.shape().to_vec(),
        })
    } else {
        Ok(base::Type::Scalar(dtype))
    }
}

/// 把 old_values 在所有节点 inputs 和图 outputs 中替换为 new_v。
/// （原节点输出 value 已无人引用，由 compact 物理删除）
fn rewrite_value_uses(graph: &mut Graph, old_values: &[ValueId], new_v: ValueId) {
    let is_old = |v: ValueId| old_values.contains(&v);
    let node_ids: Vec<u32> = graph.node_ids().collect();
    for nid in node_ids {
        let Ok(node) = graph.node(nid) else { continue; };
        let old_inputs: Vec<ValueId> = node.inputs().to_vec();
        let changed = old_inputs.iter().any(|&v| is_old(v));
        if changed {
            let new_inputs: Vec<ValueId> = old_inputs
                .iter()
                .map(|&v| if is_old(v) { new_v } else { v })
                .collect();
            graph.storage.set_node_inputs(nid, &new_inputs);
        }
    }
    let old_outputs: Vec<ValueId> = graph.outputs().to_vec();
    let changed = old_outputs.iter().any(|&v| is_old(v));
    if changed {
        let new_outputs: Vec<ValueId> = old_outputs
            .iter()
            .map(|&v| if is_old(v) { new_v } else { v })
            .collect();
        graph.storage.outputs = new_outputs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    fn tensor_type() -> Type {
        Type::Tensor {
            dtype: DType::F32,
            dims: vec![2, 3],
        }
    }

    /// 构造一个 LayerNorm 节点：inputs [x, gamma, beta]，带 Axis/Epsilon 属性
    fn build_layernorm_graph() -> Graph {
        let mut g = Graph::new("test");
        let x = g.add_input(tensor_type(), Some("x"));
        let gamma = g.add_input(tensor_type(), Some("gamma"));
        let beta = g.add_input(tensor_type(), Some("beta"));
        let ln = g.add_node(OpKind::LayerNorm);
        let out = g.add_value(tensor_type(), Some("out"), ln);
        g.storage.set_node_inputs(ln, &[x, gamma, beta]);
        g.storage.set_node_outputs(ln, &[out]);
        g.storage.add_attr_int(ln, StorageAttrKey::Axis, -1);
        g.storage.add_attr_float(ln, StorageAttrKey::Epsilon, 1e-5);
        g.mark_output(out);
        g
    }

    #[test]
    fn layernorm_decomposes_to_subgraph() {
        let mut g = build_layernorm_graph();
        let results = run_decompose(&mut g).unwrap();
        assert_eq!(results.len(), 1, "应拆分 1 个 LayerNorm");
        // 拆出 10 个细粒度节点
        assert_eq!(
            results[0].expanded.len(),
            10,
            "LayerNorm 应拆出 10 个原语节点"
        );
        // 原节点应已删除，图里不应再有 LayerNorm
        let has_ln = g
            .node_ids()
            .any(|id| g.node(id).unwrap().kind == OpKind::LayerNorm);
        assert!(!has_ln, "原 LayerNorm 节点应被删除");
        // 应有 ReduceMean ×2、ReduceMax 无、Sqrt、Div 等
        let kinds: Vec<OpKind> = g.node_ids().map(|id| g.node(id).unwrap().kind).collect();
        assert!(
            kinds.iter().filter(|k| **k == OpKind::ReduceMean).count() >= 2,
            "应有至少 2 个 ReduceMean"
        );
        assert!(kinds.contains(&OpKind::Sqrt), "应有 Sqrt");
        assert!(kinds.contains(&OpKind::Div), "应有 Div（1/std）");
    }

    #[test]
    fn softmax_decomposes_numerically_stable() {
        let mut g = Graph::new("test");
        let x = g.add_input(tensor_type(), Some("x"));
        let sm = g.add_node(OpKind::Softmax);
        let out = g.add_value(tensor_type(), Some("out"), sm);
        g.storage.set_node_inputs(sm, &[x]);
        g.storage.set_node_outputs(sm, &[out]);
        g.storage.add_attr_int(sm, StorageAttrKey::Axis, -1);
        g.mark_output(out);

        let results = run_decompose(&mut g).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].expanded.len(), 5, "Softmax 应拆出 5 个原语节点");
        // 数值稳定版必须有 ReduceMax
        let kinds: Vec<OpKind> = g.node_ids().map(|id| g.node(id).unwrap().kind).collect();
        assert!(
            kinds.contains(&OpKind::ReduceMax),
            "数值稳定 Softmax 应含 ReduceMax"
        );
        assert!(kinds.contains(&OpKind::Exp), "应有 Exp");
        assert!(kinds.contains(&OpKind::ReduceSum), "应有 ReduceSum");
        assert!(kinds.contains(&OpKind::Div), "应有 Div");
    }

    #[test]
    fn gelu_decomposes_to_tanh_approx() {
        let mut g = Graph::new("test");
        let x = g.add_input(tensor_type(), Some("x"));
        let gelu = g.add_node(OpKind::Gelu);
        let out = g.add_value(tensor_type(), Some("out"), gelu);
        g.storage.set_node_inputs(gelu, &[x]);
        g.storage.set_node_outputs(gelu, &[out]);
        g.mark_output(out);

        let results = run_decompose(&mut g).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].expanded.len(), 9, "Gelu 应拆出 9 个原语节点");
        let kinds: Vec<OpKind> = g.node_ids().map(|id| g.node(id).unwrap().kind).collect();
        assert!(kinds.contains(&OpKind::Tanh), "tanh 近似应有 Tanh");
        assert!(kinds.contains(&OpKind::Mul), "应有 Mul");
        assert!(kinds.contains(&OpKind::Add), "应有 Add");
    }

    #[test]
    fn output_rewired_to_new_subgraph() {
        // 拆分后图输出应指向新子图的最终输出，而非已删除的原节点
        let mut g = build_layernorm_graph();
        run_decompose(&mut g).unwrap();
        // 图应仍有 1 个输出
        assert_eq!(g.outputs().len(), 1);
        let out_v = g.outputs()[0];
        // 输出 value 的定义节点不应是 u32::MAX（已删除）
        let def = g.value(out_v).unwrap().def_node();
        assert_ne!(def, u32::MAX, "输出应指向新子图节点，不应是悬空 value");
        let def_node = g.node(def).unwrap();
        // LayerNorm 拆分最终输出是 Add（scaled + beta）
        assert_eq!(def_node.kind, OpKind::Add, "LayerNorm 最终输出应为 Add");
    }

    #[test]
    fn non_decomposable_nodes_untouched() {
        // 没有 LayerNorm/Softmax/Gelu 的图不应被改动
        let mut g = Graph::new("test");
        let x = g.add_input(tensor_type(), Some("x"));
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(tensor_type(), Some("out"), add);
        g.storage.set_node_inputs(add, &[x, x]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let results = run_decompose(&mut g).unwrap();
        assert!(results.is_empty(), "无可拆节点应返回空");
        assert_eq!(g.node_count(), 1, "图应未被改动");
    }
}
