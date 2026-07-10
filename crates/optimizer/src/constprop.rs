//! constprop — 常量传播（IO 同样性的补充）
//!
//! 设计哲学：识别"恒为常量的 value"，把对它的引用统一替换为单一常量节点，
//! 让后续 algebra 折叠 + CSE 合并。这是基础优化，不是模式匹配。
//!
//! 传播策略：
//! - 收集所有 Constant 节点的输出 value 及其标量值
//! - 对值相同的常量（bit pattern 相等），统一指向第一个出现的常量节点
//! - 这样两个 `add_constant_f64(2.0)` 会变成对同一常量的引用，
//!   后续 CSE 能把 `x+2` 和 `x+2`（用了不同常量节点）识别为同表达式
//!
//! 注意：本 pass 不做常量折叠（那是 algebra 的事），只做"常量 value 引用统一"。

use base::{Graph, NodeId, OpKind, Result, ValueId};
use std::collections::HashMap;

pub struct ConstPropResult {
    /// 被合并的常量节点（输出已无人引用，留给 DCE）
    pub merged_constants: Vec<NodeId>,
    /// value 替换映射：old_value → canonical_value
    pub value_replacements: HashMap<ValueId, ValueId>,
}

/// 收集常量传播结果（纯读取，不改图）
pub fn run_constprop(graph: &Graph) -> Result<ConstPropResult> {
    // bit_pattern → 第一个出现的常量 value（canonical）
    let mut canonical: HashMap<u64, ValueId> = HashMap::new();
    let mut value_replacements: HashMap<ValueId, ValueId> = HashMap::new();
    let mut merged: Vec<NodeId> = Vec::new();

    for id in graph.node_ids() {
        let n = graph.node(id)?;
        if n.kind != OpKind::Constant {
            continue;
        }
        let val = n.constant_value().unwrap_or(f64::NAN);
        let bits = val.to_bits();
        // 该常量节点的输出 value（取第一个）
        let outs = n.outputs();
        if outs.is_empty() {
            continue;
        }
        let this_val = outs[0];
        match canonical.get(&bits) {
            Some(&canon_val) => {
                // 已有同值常量，把本节点的输出 value 替换为 canonical
                for &o in outs {
                    value_replacements.insert(o, canon_val);
                }
                merged.push(id);
            }
            None => {
                canonical.insert(bits, this_val);
            }
        }
    }

    Ok(ConstPropResult {
        merged_constants: merged,
        value_replacements,
    })
}

/// 应用常量传播到图。返回被合并的常量节点数（实际删除由 DCE 完成，
/// 这里只重写引用，让重复常量节点的输出无人引用）。
pub fn apply_constprop(graph: &mut Graph) -> Result<usize> {
    let result = run_constprop(graph)?;
    if result.value_replacements.is_empty() {
        return Ok(0);
    }
    let count = result.value_replacements.len();
    rewrite_value_uses(graph, &result.value_replacements);
    Ok(count)
}

/// 重写所有节点 inputs 和图 outputs 中的 value 引用
fn rewrite_value_uses(graph: &mut Graph, replacements: &HashMap<ValueId, ValueId>) {
    let lookup = |v: ValueId| -> ValueId { replacements.get(&v).copied().unwrap_or(v) };
    let node_ids: Vec<NodeId> = graph.node_ids().collect();
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

    #[test]
    fn merges_identical_constants() {
        let mut g = Graph::new("test");
        let (_c1, a) = g.add_constant_f64(2.0);
        let (_c2, b) = g.add_constant_f64(2.0); // 同值，应被合并到 a
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let add1 = g.add_node(OpKind::Add);
        let out1 = g.add_value(Type::Scalar(DType::F32), Some("o1"), add1);
        g.storage.set_node_inputs(add1, &[x, a]);
        g.storage.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.storage.set_node_inputs(add2, &[x, b]);
        g.storage.set_node_outputs(add2, &[out2]);
        g.mark_output(out1);
        g.mark_output(out2);

        let result = run_constprop(&g).unwrap();
        assert!(
            !result.value_replacements.is_empty(),
            "同值常量应被传播合并"
        );
        // b 应被替换为 a
        assert_eq!(result.value_replacements.get(&b), Some(&a));
    }

    #[test]
    fn distinct_constants_not_merged() {
        let mut g = Graph::new("test");
        let (_c1, a) = g.add_constant_f64(1.0);
        let (_c2, b) = g.add_constant_f64(2.0);
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(Type::Scalar(DType::F32), Some("o"), add);
        g.storage.set_node_inputs(add, &[x, a, b]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let result = run_constprop(&g).unwrap();
        assert!(result.value_replacements.is_empty(), "不同值常量不应合并");
    }

    #[test]
    fn apply_rewrites_inputs() {
        let mut g = Graph::new("test");
        let (_c1, a) = g.add_constant_f64(3.0);
        let (_c2, b) = g.add_constant_f64(3.0);
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(Type::Scalar(DType::F32), Some("o"), add);
        // 用了 b（将被替换为 a）
        g.storage.set_node_inputs(add, &[x, b]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let count = apply_constprop(&mut g).unwrap();
        assert_eq!(count, 1);
        // add 的 inputs[1] 应已变成 a
        let n = g.node(add).unwrap();
        assert_eq!(n.inputs()[1], a, "b 应被替换为 a");
    }

    #[test]
    fn nan_constants_merged_consistently() {
        // NaN != NaN，但 to_bits 相同应被识别为同值常量
        let mut g = Graph::new("test");
        let (_c1, a) = g.add_constant_f64(f64::NAN);
        let (_c2, b) = g.add_constant_f64(f64::NAN);
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let add1 = g.add_node(OpKind::Add);
        let out1 = g.add_value(Type::Scalar(DType::F32), Some("o1"), add1);
        g.storage.set_node_inputs(add1, &[x, a]);
        g.storage.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.storage.set_node_inputs(add2, &[x, b]);
        g.storage.set_node_outputs(add2, &[out2]);
        g.mark_output(out1);
        g.mark_output(out2);

        let result = run_constprop(&g).unwrap();
        assert_eq!(
            result.value_replacements.get(&b),
            Some(&a),
            "NaN 的 to_bits 相同应被合并"
        );
    }
}
