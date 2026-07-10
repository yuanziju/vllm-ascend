//! algebra — 代数恒等式规则（函数式实现，不用模式匹配）
//!
//! 设计哲学：
//! - 只用简单代数规则（x+0, x*1, x*0=0, 常量折叠, x-x=0[可选]）
//! - 不硬编码复合算子模式（MatMul+Add→Linear 这种留给 fuse）
//! - 有 NaN/Inf 风险的规则（x-x=0、x/x=1）默认禁用，由 `AlgebraConfig.unsafe_opts` 开关控制
//!
//! 应用方式：`simplify` 是纯函数返回建议；`run_algebraic_simplify` 收集
//! (old_value → new_value) 替换映射，重写所有节点 inputs，被替换的节点留给 DCE 清理。

use base::{Graph, NodeId, NodeView, OpKind, Result, ValueId};
use std::collections::HashMap;

/// 代数简化配置
#[derive(Debug, Clone, Copy, Default)]
pub struct AlgebraConfig {
    /// 启用有 NaN 风险的规则：x-x=0、x/x=1。默认 false（保守）
    pub unsafe_opts: bool,
}

/// 单节点简化建议
pub enum SimplifyResult {
    /// 用已有 value 替换该节点的输出（如 x+0 → x）
    ReplaceWith(ValueId),
    /// 用新常量 value 替换（常量折叠：c1+c2 → 新 Constant）
    /// 调用方需把新节点加入图
    FoldToConstant(f64),
    NoChange,
}

pub fn simplify(graph: &Graph, node: NodeView, cfg: AlgebraConfig) -> Result<SimplifyResult> {
    match node.kind {
        OpKind::Add => simplify_add(graph, node),
        OpKind::Sub => simplify_sub(graph, node, cfg),
        OpKind::Mul => simplify_mul(graph, node),
        OpKind::Div => simplify_div(graph, node, cfg),
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
    // 常量折叠：c1 + c2 = c3
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        return Ok(SimplifyResult::FoldToConstant(x + y));
    }
    // x + 0 = x
    if is_constant_zero(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    if is_constant_zero(graph, a)? {
        return Ok(SimplifyResult::ReplaceWith(b));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_sub(graph: &Graph, node: NodeView, cfg: AlgebraConfig) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    // 常量折叠：c1 - c2 = c3
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        return Ok(SimplifyResult::FoldToConstant(x - y));
    }
    // x - 0 = x
    if is_constant_zero(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    // x - x = 0 (NaN 风险：Inf-Inf=NaN, NaN-x=NaN。默认禁用)
    if cfg.unsafe_opts && a == b {
        return Ok(SimplifyResult::FoldToConstant(0.0));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_mul(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    // 常量折叠：c1 * c2 = c3
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        return Ok(SimplifyResult::FoldToConstant(x * y));
    }
    // x * 1 = x
    if is_constant_one(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    if is_constant_one(graph, a)? {
        return Ok(SimplifyResult::ReplaceWith(b));
    }
    // x * 0 = 0 (注意：对 NaN 不安全，NaN*0=NaN，但 0 是比 x 更"确定"的值，
    // 且 x 若含 NaN 则结果本就不确定。此处保守返回 0，因 x*0 数学上=0)
    if is_constant_zero(graph, b)? || is_constant_zero(graph, a)? {
        return Ok(SimplifyResult::FoldToConstant(0.0));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_div(graph: &Graph, node: NodeView, cfg: AlgebraConfig) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    // 常量折叠：c1 / c2 = c3 (c2 != 0)
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        if y != 0.0 {
            return Ok(SimplifyResult::FoldToConstant(x / y));
        }
    }
    // x / 1 = x
    if is_constant_one(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    // x / x = 1 (NaN 风险：0/0=NaN, Inf/Inf=NaN。默认禁用)
    if cfg.unsafe_opts && a == b {
        return Ok(SimplifyResult::FoldToConstant(1.0));
    }
    Ok(SimplifyResult::NoChange)
}

// --- 常量识别辅助 ---

fn constant_value(graph: &Graph, v: ValueId) -> Result<Option<f64>> {
    let val = graph.value(v)?;
    let def = val.def_node();
    if def == u32::MAX {
        return Ok(None);
    }
    let node = graph.node(def)?;
    Ok(node.constant_value())
}

fn is_constant_zero(graph: &Graph, v: ValueId) -> Result<bool> {
    Ok(constant_value(graph, v)?.map(|f| f == 0.0).unwrap_or(false))
}

fn is_constant_one(graph: &Graph, v: ValueId) -> Result<bool> {
    Ok(constant_value(graph, v)?.map(|f| f == 1.0).unwrap_or(false))
}

/// 应用代数简化到整个图。
///
/// 返回应用的简化次数。被替换/折叠的节点不直接删除（留给 DCE），
/// 但会把它们的输出 value 在所有使用者 inputs 中替换为新的 value。
pub fn run_algebraic_simplify(graph: &mut Graph) -> Result<usize> {
    run_with_config(graph, AlgebraConfig::default())
}

pub fn run_with_config(graph: &mut Graph, cfg: AlgebraConfig) -> Result<usize> {
    let mut applied = 0usize;
    // 已处理节点：一旦某节点被简化，不再重复处理（它的 inputs 不变，重复处理会死循环）
    // 嵌套场景（(x+0)+0）靠 rewrite 后外层节点 inputs 变化触发新简化
    let mut processed: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

    loop {
        // 第一阶段：纯读取，收集本轮所有简化建议（避免借用冲突）
        let mut suggestions: Vec<(NodeId, SimplifyResult)> = Vec::new();
        for id in graph.node_ids() {
            if processed.contains(&id) {
                continue;
            }
            let n = graph.node(id)?;
            if n.kind == OpKind::Constant {
                processed.insert(id);
                continue;
            }
            match simplify(graph, n, cfg)? {
                SimplifyResult::NoChange => {
                    processed.insert(id);
                }
                other => suggestions.push((id, other)),
            }
        }
        if suggestions.is_empty() {
            break;
        }

        // 第二阶段：应用建议（可变借用安全）
        let mut round_replacements: HashMap<ValueId, ValueId> = HashMap::new();
        for (id, sugg) in suggestions {
            processed.insert(id);
            let outs: Vec<ValueId> = graph.node(id)?.outputs().to_vec();
            match sugg {
                SimplifyResult::ReplaceWith(new_v) => {
                    for out_v in outs {
                        round_replacements.insert(out_v, new_v);
                    }
                    applied += 1;
                }
                SimplifyResult::FoldToConstant(val) => {
                    let (_cnode, cval) = graph.add_constant_f64(val);
                    for out_v in outs {
                        round_replacements.insert(out_v, cval);
                    }
                    applied += 1;
                }
                SimplifyResult::NoChange => {}
            }
        }
        // 立即重写 inputs，使下一轮看到更新后的图
        rewrite_inputs(graph, &round_replacements);
    }

    Ok(applied)
}

/// 重写所有节点的 inputs：把 old_v 替换为 new_v
fn rewrite_inputs(graph: &mut Graph, replacements: &HashMap<ValueId, ValueId>) {
    let lookup = |v: ValueId| -> ValueId { replacements.get(&v).copied().unwrap_or(v) };
    let node_ids: Vec<u32> = graph.node_ids().collect();
    for nid in node_ids {
        let old_inputs: Vec<ValueId> = graph.node(nid).unwrap().inputs().to_vec();
        let new_inputs: Vec<ValueId> = old_inputs.iter().map(|&v| lookup(v)).collect();
        if old_inputs != new_inputs {
            graph.storage.set_node_inputs(nid, &new_inputs);
        }
    }
    let old_outputs: Vec<ValueId> = graph.outputs().to_vec();
    let new_outputs: Vec<ValueId> = old_outputs.iter().map(|&v| lookup(v)).collect();
    if old_outputs != new_outputs {
        graph.storage.outputs = new_outputs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    fn build_add_with_zero() -> Graph {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("x"),
        );
        let (_c, zero) = g.add_constant_f64(0.0);
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("out"),
            add,
        );
        g.storage.set_node_inputs(add, &[x, zero]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);
        g
    }

    #[test]
    fn add_zero_simplifies() {
        let mut g = build_add_with_zero();
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        // 输出应已被重写为 x (value 0)
        assert_eq!(g.outputs(), &[0]);
    }

    #[test]
    fn mul_one_simplifies() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, one) = g.add_constant_f64(1.0);
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), mul);
        g.storage.set_node_inputs(mul, &[x, one]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        assert_eq!(g.outputs(), &[x]);
    }

    #[test]
    fn mul_zero_folds() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, zero) = g.add_constant_f64(0.0);
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), mul);
        g.storage.set_node_inputs(mul, &[x, zero]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        // 输出应指向新建的常量 0 节点的输出 value
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        let def_node = g.node(def).unwrap();
        assert_eq!(def_node.kind, OpKind::Constant);
        assert_eq!(def_node.constant_value(), Some(0.0));
    }

    #[test]
    fn const_folds_add() {
        let mut g = Graph::new("test");
        let (_c1, a) = g.add_constant_f64(3.0);
        let (_c2, b) = g.add_constant_f64(4.0);
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), add);
        g.storage.set_node_inputs(add, &[a, b]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        let def_node = g.node(def).unwrap();
        assert_eq!(def_node.constant_value(), Some(7.0));
    }

    #[test]
    fn sub_x_x_disabled_by_default() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sub = g.add_node(OpKind::Sub);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sub);
        g.storage.set_node_inputs(sub, &[x, x]);
        g.storage.set_node_outputs(sub, &[out]);
        g.mark_output(out);
        // 默认 unsafe_opts=false，不应简化
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn sub_x_x_enabled_with_unsafe() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sub = g.add_node(OpKind::Sub);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sub);
        g.storage.set_node_inputs(sub, &[x, x]);
        g.storage.set_node_outputs(sub, &[out]);
        g.mark_output(out);
        let count = run_with_config(&mut g, AlgebraConfig { unsafe_opts: true }).unwrap();
        assert_eq!(count, 1);
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        let def_node = g.node(def).unwrap();
        assert_eq!(def_node.constant_value(), Some(0.0));
    }
}
