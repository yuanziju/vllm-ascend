//! float_opts — 浮点数结构优化（IEEE754 位级 trick + Flash Attention 式重排）

use base::{Graph, NodeView, OpKind, Result};

pub enum FloatOpt {
    FastInvSqrt { sqrt_node: base::NodeId, div_node: base::NodeId },
    SoftmaxOnline { softmax_node: base::NodeId },
    MulByTwoToAdd { mul_node: base::NodeId },
    DivByConstToMul { div_node: base::NodeId, reciprocal: f64 },
}

pub fn find_opportunities(graph: &Graph) -> Result<Vec<FloatOpt>> {
    let mut opts = Vec::new();
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        if n.kind == OpKind::Div {
            if let Some(opt) = try_match_fast_inv_sqrt(graph, n)? {
                opts.push(opt);
            }
            if let Some(opt) = try_match_div_by_const(graph, n)? {
                opts.push(opt);
            }
        }
        if n.kind == OpKind::Softmax {
            opts.push(FloatOpt::SoftmaxOnline { softmax_node: id });
        }
        if n.kind == OpKind::Mul {
            if let Some(opt) = try_match_mul_by_two(graph, n)? {
                opts.push(opt);
            }
        }
    }
    Ok(opts)
}

fn try_match_fast_inv_sqrt(_graph: &Graph, _div: NodeView) -> Result<Option<FloatOpt>> {
    // TODO: 接入 OpKind::Sqrt 后启用
    Ok(None)
}

fn try_match_mul_by_two(_graph: &Graph, mul: NodeView) -> Result<Option<FloatOpt>> {
    let _ = mul;
    // TODO: 接入常量值属性后启用
    Ok(None)
}

fn try_match_div_by_const(_graph: &Graph, div: NodeView) -> Result<Option<FloatOpt>> {
    let _ = div;
    // TODO: 接入常量值属性后启用
    Ok(None)
}

pub fn apply_float_opts(graph: &mut Graph) -> Result<usize> {
    let opts = find_opportunities(graph)?;
    Ok(opts.len())
}
