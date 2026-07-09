//! cse — 公共子表达式消除（IO 同样性）
//!
//! 设计哲学：识别"同样的计算"，消除冗余。指纹包含：
//! - 操作类型 (OpKind)
//! - 输入 value IDs
//! - 常量值（对 Constant 节点：值相同的常量合并为一个）
//! - 可交换操作规范化（Add/Mul 的输入排序后比较）
//!
//! 幂等性识别（relu(relu(x))=relu(x)）属于代数范畴，不在此处做。
//! CSE 只做"完全相同的子表达式"合并。

use base::{Graph, OpKind, Result, ValueId};
use std::collections::HashMap;

pub struct CseResult {
    pub removed_nodes: Vec<base::NodeId>,
    pub value_replacements: HashMap<ValueId, ValueId>,
}

/// 子表达式指纹。对常量节点，编码其标量值。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Fingerprint {
    /// 非常量节点：(op, normalized_inputs)
    /// 可交换 op (Add/Mul) 的 inputs 排序后作为指纹
    Op(u8, Vec<ValueId>),
    /// 常量节点：值相同则指纹相同（用 f64::to_bits 保证 NaN 相等性一致）
    Constant(u64),
}

fn fingerprint(graph: &Graph, node_id: base::NodeId) -> Result<Fingerprint> {
    let n = graph.node(node_id)?;
    if n.kind == OpKind::Constant {
        // 常量：用值的 bit pattern 作为指纹
        let val = n.constant_value().unwrap_or(f64::NAN);
        return Ok(Fingerprint::Constant(val.to_bits()));
    }
    let mut ins = n.inputs().to_vec();
    // 可交换操作：排序 inputs 使 a+b 和 b+a 指纹相同
    if matches!(n.kind, OpKind::Add | OpKind::Mul) {
        ins.sort_unstable();
    }
    Ok(Fingerprint::Op(n.kind as u8, ins))
}

pub fn run_cse(graph: &Graph) -> Result<CseResult> {
    let mut signatures: HashMap<Fingerprint, base::NodeId> = HashMap::new();
    let mut value_replacements: Vec<(ValueId, ValueId)> = Vec::new();
    let mut removed_nodes: Vec<base::NodeId> = Vec::new();

    for id in graph.node_ids() {
        let fp = fingerprint(graph, id)?;
        if let Some(&existing) = signatures.get(&fp) {
            // 找到等价节点，把当前节点的输出替换为 existing 的输出
            let existing_node = graph.node(existing)?;
            let current_node = graph.node(id)?;
            let existing_outputs = existing_node.outputs();
            let current_outputs = current_node.outputs();
            for (old, new) in current_outputs.iter().zip(existing_outputs.iter()) {
                value_replacements.push((*old, *new));
            }
            removed_nodes.push(id);
        } else {
            signatures.insert(fp, id);
        }
    }

    let value_replacements: HashMap<ValueId, ValueId> = value_replacements.into_iter().collect();
    Ok(CseResult {
        removed_nodes,
        value_replacements,
    })
}

pub fn apply_cse(graph: &mut Graph) -> Result<usize> {
    let result = run_cse(graph)?;
    let removed_count = result.removed_nodes.len();
    if removed_count == 0 {
        return Ok(0);
    }
    // 先重写 inputs（把被消除节点的输出替换为保留节点的输出）
    rewrite_inputs(graph, &result.value_replacements);
    // 再物理删除被消除的节点
    let remove_set: std::collections::HashSet<base::NodeId> =
        result.removed_nodes.into_iter().collect();
    let (new_graph, _, _) = graph.compact(&remove_set);
    *graph = new_graph;
    Ok(removed_count)
}

fn rewrite_inputs(graph: &mut Graph, replacements: &HashMap<ValueId, ValueId>) {
    let lookup = |v: ValueId| -> ValueId { replacements.get(&v).copied().unwrap_or(v) };
    let node_ids: Vec<u32> = graph.node_ids().collect();
    for nid in node_ids {
        let old_inputs: Vec<ValueId> = graph.node(nid).unwrap().inputs().to_vec();
        let new_inputs: Vec<ValueId> = old_inputs.iter().map(|&v| lookup(v)).collect();
        if old_inputs != new_inputs {
            graph.raw.set_node_inputs(nid, &new_inputs);
        }
    }
    let old_outputs: Vec<ValueId> = graph.outputs().to_vec();
    let new_outputs: Vec<ValueId> = old_outputs.iter().map(|&v| lookup(v)).collect();
    if old_outputs != new_outputs {
        graph.raw.outputs = new_outputs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    #[test]
    fn cse_finds_redundant_add() {
        let mut g = Graph::new("test");
        let a = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("a"),
        );
        let b = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("b"),
        );
        let add1 = g.add_node(OpKind::Add);
        let out1 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("c"),
            add1,
        );
        g.raw.set_node_inputs(add1, &[a, b]);
        g.raw.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("d"),
            add2,
        );
        g.raw.set_node_inputs(add2, &[a, b]);
        g.raw.set_node_outputs(add2, &[out2]);
        g.mark_output(out1);
        g.mark_output(out2);
        let result = run_cse(&g).unwrap();
        assert_eq!(result.removed_nodes.len(), 1);
    }

    #[test]
    fn cse_merges_identical_constants() {
        let mut g = Graph::new("test");
        // 两个值相同的常量 42.0
        let (_c1, a) = g.add_constant_f64(42.0);
        let (_c2, b) = g.add_constant_f64(42.0);
        // 用它们各做一个 add（其实可以合并常量）
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let add1 = g.add_node(OpKind::Add);
        let out1 = g.add_value(Type::Scalar(DType::F32), Some("o1"), add1);
        g.raw.set_node_inputs(add1, &[x, a]);
        g.raw.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.raw.set_node_inputs(add2, &[x, b]);
        g.raw.set_node_outputs(add2, &[out2]);
        g.mark_output(out1);
        g.mark_output(out2);
        // CSE 应识别两个 42.0 常量指纹相同，合并之
        let result = run_cse(&g).unwrap();
        assert!(
            !result.removed_nodes.is_empty(),
            "应至少合并一个常量节点, got {}",
            result.removed_nodes.len()
        );
    }

    #[test]
    fn cse_commutative_normalizes() {
        let mut g = Graph::new("test");
        let a = g.add_input(Type::Scalar(DType::F32), Some("a"));
        let b = g.add_input(Type::Scalar(DType::F32), Some("b"));
        // a + b
        let add1 = g.add_node(OpKind::Add);
        let out1 = g.add_value(Type::Scalar(DType::F32), Some("o1"), add1);
        g.raw.set_node_inputs(add1, &[a, b]);
        g.raw.set_node_outputs(add1, &[out1]);
        // b + a （顺序不同，但可交换，应被 CSE 识别）
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.raw.set_node_inputs(add2, &[b, a]);
        g.raw.set_node_outputs(add2, &[out2]);
        g.mark_output(out1);
        g.mark_output(out2);
        let result = run_cse(&g).unwrap();
        assert_eq!(
            result.removed_nodes.len(),
            1,
            "a+b 和 b+a 应被识别为同一表达式"
        );
    }

    #[test]
    fn cse_distinct_constants_not_merged() {
        let mut g = Graph::new("test");
        let (_c1, a) = g.add_constant_f64(1.0);
        let (_c2, b) = g.add_constant_f64(2.0);
        let add1 = g.add_node(OpKind::Add);
        let out1 = g.add_value(Type::Scalar(DType::F32), Some("o1"), add1);
        g.raw.set_node_inputs(add1, &[a, a]);
        g.raw.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.raw.set_node_inputs(add2, &[b, b]);
        g.raw.set_node_outputs(add2, &[out2]);
        g.mark_output(out1);
        g.mark_output(out2);
        // 1.0 和 2.0 指纹不同，不应合并
        let result = run_cse(&g).unwrap();
        assert_eq!(result.removed_nodes.len(), 0);
    }
}
