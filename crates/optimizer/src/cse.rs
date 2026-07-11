//! cse — 公共子表达式消除（IO 同样性）
//!
//! 设计哲学：识别"同样的计算"，消除冗余。指纹包含：
//! - 操作类型 (OpKind)
//! - 输入 value IDs
//! - 属性哈希（Axis/Perm/Shape 等，带 attr 的 op 不再误合并）
//! - 常量值（单元素常量按值合并；多元素张量按 dims+values 合并）
//! - 可交换操作规范化（Add/Mul 的输入排序后比较）
//!
//! 幂等性识别（relu(relu(x))=relu(x)）属于代数范畴，不在此处做。
//! CSE 只做"完全相同的子表达式"合并。CsePass::run 含不动点迭代，
//! 消除后暴露的新机会能多捕一层。

use base::{Graph, OpKind, Result, ValueId};
use std::collections::HashMap;

pub struct CseResult {
    pub removed_nodes: Vec<base::NodeId>,
    pub value_replacements: HashMap<ValueId, ValueId>,
}

/// 可交换二元操作：a·b = b·a，CSE 时排序 inputs 使交换前后的写法指纹相同。
/// 未来新增可交换 op（如 Max/Min/Equal）在此集中维护，避免散落在指纹逻辑里
fn is_commutative(kind: OpKind) -> bool {
    matches!(kind, OpKind::Add | OpKind::Mul)
}

/// 子表达式指纹。CSE 据 IO 同样性识别公共子表达式。
///
/// 指纹包含：
/// - 操作类型 (OpKind)
/// - 输入 value IDs（可交换 op 排序后比较）
/// - **属性哈希** (attr_hash)：Reduce/Concat/Transpose/Reshape/Slice/Fused 等 op 的
///   语义由 attr 决定，inputs 相同但 attr 不同时是不同计算，必须区分
/// - 常量值（单元素走 Constant，多元素张量走 ConstantTensor，未知常量不参与 CSE）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Fingerprint {
    /// 非常量节点：(op, normalized_inputs, attr_hash)
    /// attr_hash 为 0 表示无 attr 或 attr 不影响语义
    Op(u8, Vec<ValueId>, u64),
    /// 单元素常量：值相同则指纹相同（用 f64::to_bits 保证 NaN 相等性一致）
    Constant(u64),
    /// 多元素常量张量：(dims, values) 的哈希——值与 shape 都相同才合并
    ConstantTensor(u64),
}

/// 计算节点属性的指纹哈希。遍历所有 attr，把 (key, tag, value) 累加成 u64。
/// 同一节点的 attr 顺序稳定（按插入序），两节点 attr 内容相同则哈希相同。
fn attr_fingerprint(n: base::NodeView<'_>) -> u64 {
    let mut h: u64 = 0;
    for e in n.attrs() {
        // 累加 (key, tag, value_hash)
        let mut entry_hash: u64 = (e.key as u64).wrapping_mul(31) ^ (e.tag as u64);
        match base::storage::AttrTag::from_u8(e.tag) {
            Some(base::storage::AttrTag::Int) => {
                entry_hash ^= n.storage.attr_int(e) as u64;
            }
            Some(base::storage::AttrTag::Float) => {
                entry_hash ^= n.storage.attr_float(e).to_bits();
            }
            Some(base::storage::AttrTag::Bool) => {
                entry_hash ^= n.storage.attr_bool(e) as u64;
            }
            Some(base::storage::AttrTag::IntArray) => {
                for &v in n.storage.attr_int_array(e) {
                    entry_hash = entry_hash.wrapping_mul(31).wrapping_add(v as u64);
                }
            }
            Some(base::storage::AttrTag::FloatArray) => {
                for &v in n.storage.attr_float_array(e) {
                    entry_hash = entry_hash.wrapping_mul(31).wrapping_add(v.to_bits());
                }
            }
            None => {}
        }
        h = h.wrapping_mul(37).wrapping_add(entry_hash);
    }
    h
}

/// 计算节点指纹。返回 None 表示该节点不参与 CSE（如非 FLOAT 的未知常量）。
fn fingerprint(graph: &Graph, node_id: base::NodeId) -> Result<Option<Fingerprint>> {
    let n = graph.node(node_id)?;
    if n.kind == OpKind::Constant {
        // 单元素常量（标量 Float 或单元素 FloatArray）
        if let Some(val) = n.constant_value() {
            return Ok(Some(Fingerprint::Constant(val.to_bits())));
        }
        // 多元素常量张量（FloatArray）：哈希 dims + values 防止不同 shape/值误合并
        if let Some(tensor) = n.constant_tensor() {
            let shape: Vec<i64> = n
                .outputs()
                .first()
                .and_then(|&vid| graph.value(vid).ok())
                .map(|v| v.shape().to_vec())
                .unwrap_or_default();
            let mut h: u64 = 0;
            for &d in &shape {
                h = h.wrapping_mul(31).wrapping_add(d as u64);
            }
            for &v in tensor {
                h = h.wrapping_mul(37).wrapping_add(v.to_bits());
            }
            return Ok(Some(Fingerprint::ConstantTensor(h)));
        }
        // 未知常量（非 FLOAT 张量等），保守不参与 CSE
        return Ok(None);
    }
    let mut ins = n.inputs().to_vec();
    // 可交换操作：排序 inputs 使 a+b 和 b+a 指纹相同
    if is_commutative(n.kind) {
        ins.sort_unstable();
    }
    let attr_hash = attr_fingerprint(n);
    Ok(Some(Fingerprint::Op(n.kind as u8, ins, attr_hash)))
}

pub fn run_cse(graph: &Graph) -> Result<CseResult> {
    let mut signatures: HashMap<Fingerprint, base::NodeId> = HashMap::new();
    let mut value_replacements: Vec<(ValueId, ValueId)> = Vec::new();
    let mut removed_nodes: Vec<base::NodeId> = Vec::new();

    for id in graph.node_ids() {
        // None 表示该节点不参与 CSE（如未知常量），跳过
        let Some(fp) = fingerprint(graph, id)? else {
            continue;
        };
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
        g.storage.set_node_inputs(add1, &[a, b]);
        g.storage.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("d"),
            add2,
        );
        g.storage.set_node_inputs(add2, &[a, b]);
        g.storage.set_node_outputs(add2, &[out2]);
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
        g.storage.set_node_inputs(add1, &[x, a]);
        g.storage.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.storage.set_node_inputs(add2, &[x, b]);
        g.storage.set_node_outputs(add2, &[out2]);
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
        g.storage.set_node_inputs(add1, &[a, b]);
        g.storage.set_node_outputs(add1, &[out1]);
        // b + a （顺序不同，但可交换，应被 CSE 识别）
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.storage.set_node_inputs(add2, &[b, a]);
        g.storage.set_node_outputs(add2, &[out2]);
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
        g.storage.set_node_inputs(add1, &[a, a]);
        g.storage.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(Type::Scalar(DType::F32), Some("o2"), add2);
        g.storage.set_node_inputs(add2, &[b, b]);
        g.storage.set_node_outputs(add2, &[out2]);
        g.mark_output(out1);
        g.mark_output(out2);
        // 1.0 和 2.0 指纹不同，不应合并
        let result = run_cse(&g).unwrap();
        assert_eq!(result.removed_nodes.len(), 0);
    }

    // --- 缺口 A 回归：带 attr 的 op 不能因 inputs 相同就误合并 ---

    #[test]
    fn cse_reduce_different_axis_not_merged() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        // ReduceMean axis=0
        let r1 = g.add_node(OpKind::ReduceMean);
        let o1 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![3],
            },
            Some("o1"),
            r1,
        );
        g.storage.set_node_inputs(r1, &[x]);
        g.storage.set_node_outputs(r1, &[o1]);
        g.storage.add_attr_int(r1, base::StorageAttrKey::Axis, 0);
        // ReduceMean axis=1（同输入 x，但 axis 不同 → 不同计算）
        let r2 = g.add_node(OpKind::ReduceMean);
        let o2 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2],
            },
            Some("o2"),
            r2,
        );
        g.storage.set_node_inputs(r2, &[x]);
        g.storage.set_node_outputs(r2, &[o2]);
        g.storage.add_attr_int(r2, base::StorageAttrKey::Axis, 1);
        g.mark_output(o1);
        g.mark_output(o2);
        let result = run_cse(&g).unwrap();
        assert_eq!(
            result.removed_nodes.len(),
            0,
            "不同 axis 的 reduce 是不同计算，不应合并"
        );
    }

    #[test]
    fn cse_transpose_different_perm_not_merged() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        // Transpose perm=[0,1]（单位排列）
        let t1 = g.add_node(OpKind::Transpose);
        let o1 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("o1"),
            t1,
        );
        g.storage.set_node_inputs(t1, &[x]);
        g.storage.set_node_outputs(t1, &[o1]);
        g.storage
            .add_attr_int_array(t1, base::StorageAttrKey::Perm, &[0, 1]);
        // Transpose perm=[1,0]（转置，同输入但 perm 不同 → 不同计算）
        let t2 = g.add_node(OpKind::Transpose);
        let o2 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![3, 2],
            },
            Some("o2"),
            t2,
        );
        g.storage.set_node_inputs(t2, &[x]);
        g.storage.set_node_outputs(t2, &[o2]);
        g.storage
            .add_attr_int_array(t2, base::StorageAttrKey::Perm, &[1, 0]);
        g.mark_output(o1);
        g.mark_output(o2);
        let result = run_cse(&g).unwrap();
        assert_eq!(
            result.removed_nodes.len(),
            0,
            "不同 perm 的 transpose 是不同计算，不应合并"
        );
    }

    #[test]
    fn cse_reduce_same_axis_merged() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let r1 = g.add_node(OpKind::ReduceMean);
        let o1 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![3],
            },
            Some("o1"),
            r1,
        );
        g.storage.set_node_inputs(r1, &[x]);
        g.storage.set_node_outputs(r1, &[o1]);
        g.storage.add_attr_int(r1, base::StorageAttrKey::Axis, 1);
        // 第二个 ReduceMean 同输入同 axis → 应合并
        let r2 = g.add_node(OpKind::ReduceMean);
        let o2 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![3],
            },
            Some("o2"),
            r2,
        );
        g.storage.set_node_inputs(r2, &[x]);
        g.storage.set_node_outputs(r2, &[o2]);
        g.storage.add_attr_int(r2, base::StorageAttrKey::Axis, 1);
        g.mark_output(o1);
        g.mark_output(o2);
        let result = run_cse(&g).unwrap();
        assert_eq!(
            result.removed_nodes.len(),
            1,
            "同输入同 axis 的 reduce 应合并"
        );
    }

    // --- 缺口 B 回归：多元素常量张量指纹不能塌缩 ---

    /// 构造一个多元素 Constant 张量节点（Value=FloatArray），shape=dims，值=vals。
    fn add_constant_tensor(g: &mut Graph, dims: &[i64], vals: &[f64], name: &str) -> ValueId {
        let node = g.add_node(OpKind::Constant);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: dims.to_vec(),
            },
            Some(name),
            node,
        );
        g.storage.set_node_outputs(node, &[out]);
        g.storage
            .add_attr_float_array(node, base::StorageAttrKey::Value, vals);
        out
    }

    #[test]
    fn cse_distinct_multielement_constants_not_merged() {
        let mut g = Graph::new("test");
        let a = add_constant_tensor(&mut g, &[3], &[1.0, 2.0, 3.0], "a");
        let b = add_constant_tensor(&mut g, &[3], &[4.0, 5.0, 6.0], "b");
        g.mark_output(a);
        g.mark_output(b);
        let result = run_cse(&g).unwrap();
        assert_eq!(
            result.removed_nodes.len(),
            0,
            "不同值的多元素常量张量不应合并"
        );
    }

    #[test]
    fn cse_identical_multielement_constants_merged() {
        let mut g = Graph::new("test");
        let a = add_constant_tensor(&mut g, &[3], &[1.0, 2.0, 3.0], "a");
        let b = add_constant_tensor(&mut g, &[3], &[1.0, 2.0, 3.0], "b");
        g.mark_output(a);
        g.mark_output(b);
        let result = run_cse(&g).unwrap();
        assert_eq!(
            result.removed_nodes.len(),
            1,
            "相同值相同 shape 的多元素常量张量应合并"
        );
    }

    // --- 缺口 C：不动点迭代，消除后暴露的新机会能多捕一层 ---

    #[test]
    fn cse_fixpoint_catches_newly_exposed() {
        // 构图：
        //   a = Add(x, y)
        //   b = Add(x, y)   ← 第一轮消除，b 的使用者 c 的输入 b→a
        //   c = Add(a, b)   ← 第一轮后变 Add(a, a)
        //   d = Add(a, a)   ← 第一轮前就存在，与第一轮后的 c 指纹相同
        // 单次 CSE 只消除 b，不动点能再消除 c 或 d 之一。
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let y = g.add_input(Type::Scalar(DType::F32), Some("y"));

        let a = g.add_node(OpKind::Add);
        let a_out = g.add_value(Type::Scalar(DType::F32), Some("a"), a);
        g.storage.set_node_inputs(a, &[x, y]);
        g.storage.set_node_outputs(a, &[a_out]);

        let b = g.add_node(OpKind::Add);
        let b_out = g.add_value(Type::Scalar(DType::F32), Some("b"), b);
        g.storage.set_node_inputs(b, &[x, y]);
        g.storage.set_node_outputs(b, &[b_out]);

        let c = g.add_node(OpKind::Add);
        let c_out = g.add_value(Type::Scalar(DType::F32), Some("c"), c);
        g.storage.set_node_inputs(c, &[a_out, b_out]);
        g.storage.set_node_outputs(c, &[c_out]);

        let d = g.add_node(OpKind::Add);
        let d_out = g.add_value(Type::Scalar(DType::F32), Some("d"), d);
        g.storage.set_node_inputs(d, &[a_out, a_out]);
        g.storage.set_node_outputs(d, &[d_out]);

        g.mark_output(c_out);
        g.mark_output(d_out);

        // 单次 apply_cse 只消除 b（1 个），不动点应消除 2 个（b + c/d 之一）
        let n1 = apply_cse(&mut g).unwrap();
        assert_eq!(n1, 1, "第一轮应消除 b");
        // 第二轮：c 的输入已 b→a，c=Add(a,a) 与 d=Add(a,a) 指纹相同，再消除 1 个
        let n2 = apply_cse(&mut g).unwrap();
        assert_eq!(n2, 1, "第二轮应消除 c 或 d（不动点暴露的新机会）");
        // 第三轮应无变化
        let n3 = apply_cse(&mut g).unwrap();
        assert_eq!(n3, 0, "第三轮应收敛");
    }
}
