//! shape_infer — 形状推断 pass
//!
//! 设计哲学：依赖类型系统要求 shape 进入类型，但前端解析时 shape 往往未知
//! （标记 -1）。本 pass 按拓扑顺序回填每个节点输出的 shape，让后续 cost_model
//! 估算准确（否则 value_bytes 会退化成 1）。
//!
//! 推断规则（保守，遇到未知或无法确定的就保留原 shape）：
//! - **elementwise**（Add/Sub/Mul/Div/Relu/Gelu/Sigmoid/Tanh/Sqrt/Rsqrt/Exp/Pow）：
//!   单输入取输入 shape；双输入取广播结果 shape
//! - **reduce**（ReduceSum/ReduceMean/ReduceMax）：沿 axis 消去一维
//! - **MatMul**：取第一个输入的行 × 第二个输入的列
//! - **Reshape**：输出 shape = attr Shape（IntArray）
//! - **Transpose**：输出 shape = 输入 shape 按 perm 重排
//! - **Concat**：沿 axis 拼接，输出 shape = 各输入 shape 在 axis 维求和，其余维相等
//! - 其余 op 保守不动
//!
//! 未知标记：shape 含 -1 表示该维未知。广播时 -1 维向已知维靠拢。

use base::{Graph, NodeId, OpKind, Result};

/// 判断 shape 是否全已知（无 -1）
fn shape_known(s: &[i64]) -> bool {
    !s.is_empty() && s.iter().all(|&d| d > 0)
}

/// 两个 shape 广播结果。规则：右对齐，每维取较大值；-1 向已知靠拢。
/// 全未知返回 None（保留原 shape）。
fn broadcast(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let av = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let bv = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        let v = match (av, bv) {
            (-1, -1) => -1,
            (-1, x) => x,
            (x, -1) => x,
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None, // 不兼容广播，保守放弃
        };
        out.push(v);
    }
    out.reverse();
    Some(out)
}

/// 推断单个节点输出的 shape（返回 None 表示保守不动）
fn infer_shape(graph: &Graph, node_id: NodeId) -> Result<Option<Vec<i64>>> {
    let n = graph.node(node_id)?;
    let kind = n.kind;
    let ins = n.inputs();

    let inferred = match kind {
        // 单输入 elementwise：输出 = 输入（要求输入 shape 已知）
        OpKind::Relu
        | OpKind::Gelu
        | OpKind::Sigmoid
        | OpKind::Tanh
        | OpKind::Sqrt
        | OpKind::Rsqrt
        | OpKind::Exp => {
            if ins.is_empty() {
                return Ok(None);
            }
            let s = graph.value(ins[0])?.shape().to_vec();
            if s.is_empty() || !shape_known(&s) {
                return Ok(None);
            }
            s
        }
        // 双输入 elementwise：广播（要求两个输入 shape 都已知，否则保守不推，
        // 避免用未知输入推出错误 shape 然后被锁定）
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Pow => {
            if ins.len() < 2 {
                return Ok(None);
            }
            let a = graph.value(ins[0])?.shape();
            let b = graph.value(ins[1])?.shape();
            if !shape_known(a) || !shape_known(b) {
                return Ok(None);
            }
            match broadcast(a, b) {
                Some(s) => s,
                None => return Ok(None),
            }
        }
        // reduce：沿 axis 消去一维（要求输入 shape 已知）
        OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax => {
            if ins.is_empty() {
                return Ok(None);
            }
            let s = graph.value(ins[0])?.shape().to_vec();
            if s.is_empty() || !shape_known(&s) {
                return Ok(None);
            }
            let axis = read_axis(graph, node_id)?;
            let mut out = Vec::with_capacity(s.len().saturating_sub(1));
            for (i, &d) in s.iter().enumerate() {
                let ax = if axis < 0 {
                    axis + s.len() as i64
                } else {
                    axis
                };
                if i as i64 != ax {
                    out.push(d);
                }
            }
            out
        }
        // MatMul：[m,k] × [k,n] → [m,n]。保守要求双输入 rank>=2 且 shape 已知
        OpKind::MatMul => {
            if ins.len() < 2 {
                return Ok(None);
            }
            let a = graph.value(ins[0])?.shape();
            let b = graph.value(ins[1])?.shape();
            if a.len() < 2 || b.len() < 2 {
                return Ok(None);
            }
            if !shape_known(a) || !shape_known(b) {
                return Ok(None);
            }
            let m = a[a.len() - 2];
            let n = b[b.len() - 1];
            // 简化：只推最后一维 shape [m, n]，忽略 batch
            vec![m, n]
        }
        // Reshape：输出 shape = attr Shape（IntArray），要求输入 shape 已知
        OpKind::Reshape => {
            if ins.is_empty() {
                return Ok(None);
            }
            let s = graph.value(ins[0])?.shape();
            if !shape_known(s) {
                return Ok(None);
            }
            match read_int_array_attr(graph, node_id, base::StorageAttrKey::Shape)? {
                Some(target) => target,
                None => return Ok(None),
            }
        }
        // Transpose：输出 shape = 输入 shape 按 perm 重排，要求输入 shape 已知
        OpKind::Transpose => {
            if ins.is_empty() {
                return Ok(None);
            }
            let s = graph.value(ins[0])?.shape().to_vec();
            if s.is_empty() || !shape_known(&s) {
                return Ok(None);
            }
            match read_int_array_attr(graph, node_id, base::StorageAttrKey::Perm)? {
                Some(perm) => {
                    if perm.len() != s.len() {
                        return Ok(None);
                    }
                    let mut out = Vec::with_capacity(s.len());
                    for &p in &perm {
                        if p < 0 || p as usize >= s.len() {
                            return Ok(None);
                        }
                        out.push(s[p as usize]);
                    }
                    out
                }
                None => return Ok(None),
            }
        }
        // Concat：沿 axis 拼接，输出 shape = 各输入 shape 在 axis 维求和，其余维相等
        OpKind::Concat => {
            if ins.is_empty() {
                return Ok(None);
            }
            // 收集所有输入 shape，要求全部已知
            let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(ins.len());
            for &vin in ins {
                let s = graph.value(vin)?.shape().to_vec();
                if s.is_empty() || !shape_known(&s) {
                    return Ok(None);
                }
                shapes.push(s);
            }
            let axis = read_axis(graph, node_id)?;
            let first = &shapes[0];
            let rank = first.len();
            let ax = if axis < 0 { axis + rank as i64 } else { axis };
            if ax < 0 || ax as usize >= rank {
                return Ok(None);
            }
            // 各输入 rank 相同，且非 axis 维相等；axis 维求和
            let mut sum_axis = first[ax as usize];
            for s in &shapes[1..] {
                if s.len() != rank {
                    return Ok(None);
                }
                for i in 0..rank {
                    if i as i64 == ax {
                        sum_axis += s[i];
                    } else if s[i] != first[i] {
                        return Ok(None);
                    }
                }
            }
            let mut out = first.clone();
            out[ax as usize] = sum_axis;
            out
        }
        // 其余 op 保守不动
        _ => return Ok(None),
    };

    Ok(Some(inferred))
}

/// 读取节点 Axis 属性（Int），默认 -1
fn read_axis(graph: &Graph, node_id: NodeId) -> Result<i64> {
    let n = graph.node(node_id)?;
    for e in n.attrs() {
        if e.key == base::StorageAttrKey::Axis as u8 && e.tag == base::storage::AttrTag::Int as u8 {
            return Ok(n.storage.attr_int(e));
        }
    }
    Ok(-1)
}

/// 读取节点 IntArray 属性，找不到返回 None
fn read_int_array_attr(
    graph: &Graph,
    node_id: NodeId,
    key: base::StorageAttrKey,
) -> Result<Option<Vec<i64>>> {
    let n = graph.node(node_id)?;
    for e in n.attrs() {
        if e.key == key as u8 && e.tag == base::storage::AttrTag::IntArray as u8 {
            return Ok(Some(n.storage.attr_int_array(e).to_vec()));
        }
    }
    Ok(None)
}

/// 应用形状推断到整个图。返回回填的 value 数。
/// 多轮迭代直到不动点（前驱 shape 回填后，后继才能推断）。
/// 不用永久 visited 集合——每轮重新评估所有节点，shape_known 检查防止重复回填已知 shape。
pub fn apply_shape_infer(graph: &mut Graph) -> Result<usize> {
    let mut filled = 0usize;

    // 多轮迭代直到不动点
    let mut changed = true;
    while changed {
        changed = false;
        // 第一阶段：纯读取，收集本轮要回填的 (out_value, new_shape)
        let mut to_fill: Vec<(base::ValueId, Vec<i64>)> = Vec::new();
        for id in graph.node_ids() {
            // 推断（纯读取）
            let inferred = infer_shape(graph, id)?;
            let new_shape = match inferred {
                Some(s) => s,
                None => continue,
            };
            let n = graph.node(id)?;
            for &out_v in n.outputs() {
                let old = graph.value(out_v)?.shape().to_vec();
                if !shape_known(&old) && shape_known(&new_shape) {
                    to_fill.push((out_v, new_shape.clone()));
                }
            }
        }
        // 第二阶段：应用回填（可变借用安全）
        for (out_v, new_shape) in to_fill {
            graph.storage.set_value_shape(out_v, &new_shape);
            filled += 1;
            changed = true;
        }
    }

    Ok(filled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, StorageAttrKey, Type};

    fn tensor(dims: Vec<i64>) -> Type {
        Type::Tensor {
            dtype: DType::F32,
            dims,
        }
    }

    #[test]
    fn infers_elementwise_passthrough() {
        let mut g = Graph::new("test");
        let x = g.add_input(tensor(vec![2, 3]), Some("x"));
        let relu = g.add_node(OpKind::Relu);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), relu); // 未知 shape
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[out]);
        g.mark_output(out);

        let n = apply_shape_infer(&mut g).unwrap();
        assert!(n >= 1, "应回填至少 1 个 value");
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[2, 3], "Relu 输出 shape 应等于输入");
    }

    #[test]
    fn infers_binary_broadcast() {
        let mut g = Graph::new("test");
        let a = g.add_input(tensor(vec![2, 3]), Some("a"));
        let b = g.add_input(tensor(vec![1, 3]), Some("b"));
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), add);
        g.storage.set_node_inputs(add, &[a, b]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[2, 3], "[2,3]+[1,3] 广播为 [2,3]");
    }

    #[test]
    fn infers_reduce_axis() {
        let mut g = Graph::new("test");
        let x = g.add_input(tensor(vec![2, 3, 4]), Some("x"));
        let rs = g.add_node(OpKind::ReduceSum);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), rs);
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[out]);
        g.storage.add_attr_int(rs, StorageAttrKey::Axis, 1);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[2, 4], "沿 axis=1 reduce 后应消去该维");
    }

    #[test]
    fn infers_matmul() {
        let mut g = Graph::new("test");
        let a = g.add_input(tensor(vec![4, 8]), Some("a"));
        let b = g.add_input(tensor(vec![8, 16]), Some("b"));
        let mm = g.add_node(OpKind::MatMul);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), mm);
        g.storage.set_node_inputs(mm, &[a, b]);
        g.storage.set_node_outputs(mm, &[out]);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[4, 16], "[4,8]×[8,16] = [4,16]");
    }

    #[test]
    fn known_shape_not_overwritten() {
        // 已知 shape 不应被覆盖
        let mut g = Graph::new("test");
        let x = g.add_input(tensor(vec![2, 3]), Some("x"));
        let relu = g.add_node(OpKind::Relu);
        let out = g.add_value(tensor(vec![9, 9]), Some("o"), relu); // 已知但故意写错
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[out]);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[9, 9], "已知 shape 不应被覆盖");
    }

    #[test]
    fn chain_inference() {
        // 链式：relu(x) -> add(., b)，两步都应回填
        let mut g = Graph::new("test");
        let x = g.add_input(tensor(vec![2, 3]), Some("x"));
        let b = g.add_input(tensor(vec![1, 3]), Some("b"));
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(tensor(vec![-1, -1]), Some("r"), relu);
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[r_out]);
        let add = g.add_node(OpKind::Add);
        let a_out = g.add_value(tensor(vec![-1, -1]), Some("a"), add);
        g.storage.set_node_inputs(add, &[r_out, b]);
        g.storage.set_node_outputs(add, &[a_out]);
        g.mark_output(a_out);

        let n = apply_shape_infer(&mut g).unwrap();
        assert!(n >= 2, "链式两步都应回填");
        assert_eq!(g.value(a_out).unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn infers_reshape() {
        let mut g = Graph::new("test");
        let x = g.add_input(tensor(vec![2, 3]), Some("x"));
        let rs = g.add_node(OpKind::Reshape);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), rs);
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[out]);
        g.storage
            .add_attr_int_array(rs, StorageAttrKey::Shape, &[6]);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[6], "Reshape 输出 shape = attr Shape");
    }

    #[test]
    fn infers_transpose() {
        let mut g = Graph::new("test");
        let x = g.add_input(tensor(vec![2, 3]), Some("x"));
        let tr = g.add_node(OpKind::Transpose);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), tr);
        g.storage.set_node_inputs(tr, &[x]);
        g.storage.set_node_outputs(tr, &[out]);
        g.storage
            .add_attr_int_array(tr, StorageAttrKey::Perm, &[1, 0]);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[3, 2], "Transpose perm=[1,0] 把 [2,3] 重排为 [3,2]");
    }

    #[test]
    fn infers_concat_along_axis() {
        let mut g = Graph::new("test");
        // 三个输入 [2,3] [2,5] [2,4] 沿 axis=1 拼接 → [2,12]
        let a = g.add_input(tensor(vec![2, 3]), Some("a"));
        let b = g.add_input(tensor(vec![2, 5]), Some("b"));
        let c = g.add_input(tensor(vec![2, 4]), Some("c"));
        let cat = g.add_node(OpKind::Concat);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), cat);
        g.storage.set_node_inputs(cat, &[a, b, c]);
        g.storage.set_node_outputs(cat, &[out]);
        g.storage.add_attr_int(cat, StorageAttrKey::Axis, 1);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[2, 12], "Concat axis=1 [2,3]+[2,5]+[2,4] → [2,12]");
    }

    #[test]
    fn concat_non_matching_dims_not_inferred() {
        let mut g = Graph::new("test");
        // 非轴维不匹配：[2,3] + [4,5] axis=1 → 第二维可拼接但第一维不等，应保守不推
        let a = g.add_input(tensor(vec![2, 3]), Some("a"));
        let b = g.add_input(tensor(vec![4, 5]), Some("b"));
        let cat = g.add_node(OpKind::Concat);
        let out = g.add_value(tensor(vec![-1, -1]), Some("o"), cat);
        g.storage.set_node_inputs(cat, &[a, b]);
        g.storage.set_node_outputs(cat, &[out]);
        g.storage.add_attr_int(cat, StorageAttrKey::Axis, 1);
        g.mark_output(out);

        apply_shape_infer(&mut g).unwrap();
        // 非轴维不匹配，应保留 [-1,-1] 不推
        let s = g.value(out).unwrap().shape();
        assert_eq!(s, &[-1i64, -1], "非轴维不匹配时应保守不推");
    }
}
