//! algebra — 代数恒等式规则（函数式实现）

use base::{Graph, NodeView, OpKind, Result, ValueId};

pub enum SimplifyResult {
    ReplaceWith(ValueId),
    NoChange,
}

pub fn simplify(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    match node.kind {
        OpKind::Add => simplify_add(graph, node),
        OpKind::Sub => simplify_sub(graph, node),
        OpKind::Mul => simplify_mul(graph, node),
        _ => Ok(SimplifyResult::NoChange),
    }
}

fn simplify_add(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    if is_constant_zero(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    if is_constant_zero(graph, a)? {
        return Ok(SimplifyResult::ReplaceWith(b));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_sub(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    if is_constant_zero(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    // x - x = 0 (保守：NaN/Inf 风险，暂不启用)
    Ok(SimplifyResult::NoChange)
}

fn simplify_mul(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    if is_constant_one(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    if is_constant_one(graph, a)? {
        return Ok(SimplifyResult::ReplaceWith(b));
    }
    Ok(SimplifyResult::NoChange)
}

fn is_constant_zero(graph: &Graph, v: ValueId) -> Result<bool> {
    is_constant_with(graph, v, |f| f == 0.0)
}

fn is_constant_one(graph: &Graph, v: ValueId) -> Result<bool> {
    is_constant_with(graph, v, |f| f == 1.0)
}

fn is_constant_with<F: Fn(f64) -> bool>(graph: &Graph, v: ValueId, pred: F) -> Result<bool> {
    let val = graph.value(v)?;
    let def = val.def_node();
    if def == u32::MAX {
        return Ok(false);
    }
    let node = graph.node(def)?;
    if node.kind != OpKind::Constant {
        return Ok(false);
    }
    let _ = pred;
    Ok(false) // TODO: 接入常量值属性后启用
}

pub fn run_algebraic_simplify(graph: &mut Graph) -> Result<usize> {
    let mut simplified = 0usize;
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        match simplify(graph, n)? {
            SimplifyResult::ReplaceWith(_) => {
                simplified += 1;
            }
            SimplifyResult::NoChange => {}
        }
    }
    Ok(simplified)
}
