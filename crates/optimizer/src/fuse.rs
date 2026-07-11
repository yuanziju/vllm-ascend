//! fuse — 多对一启发式融合（基于 cost model）
//!
//! 设计哲学：把可融合的算子链合成单个 Fused 节点，省 launch overhead +
//! 中间结果访存。用 cost_model 判定融合收益（只融合收益 > 0 的链）。
//!
//! 融合策略：
//! - elementwise 链（Add/Sub/Mul/Div/Relu/Sigmoid/... 单输出算子链）
//! - 链头允许一个 reduce（reduce 改变 shape，是 shape 分界点，不再往前扩）
//! - binary elementwise（Add/Sub/Mul/Div）的"另一输入"作为 side input 收集，
//!   融合后节点 inputs = 链头 inputs + 各 binary 的 side inputs（按链序）
//! - 链尾节点 op 改成 Fused，attr 记录 op 序列 + side input 位置
//! - 链中其余节点变死代码，由 DCE 清理
//!
//! **Fused vs Custom**：Fused 专管融合产物（本 pass 产生），Custom 留给未知 ONNX 算子
//! （frontend 产生）。lowering 按 op_kind 直接分派，不靠 attr 探测猜语义。
//!
//! 属性编码（供 lowering 重建）：
//! - `Shape`（IntArray）：op 序列 [op0, op1, ...]（每个是 OpKind as u8）
//! - `Strides`（IntArray）：side input 位置 [pos0, pos1, ...]（每个是该节点
//!   inputs 中 side input 的下标；-1 表示该节点无 side input，即 unary/链头）

use crate::cost_model::{fusion_saving, CostCoeffs};
use base::{Graph, OpKind, Result, ValueId};

pub struct FusionOpportunity {
    pub nodes: Vec<base::NodeId>,
    /// 各 binary 节点的 side input（按链 head→tail 序），融合后追加到节点 inputs
    pub side_inputs: Vec<ValueId>,
    /// 每个链节点在其 inputs 中 side input 的下标（-1 = 无 side input）
    pub side_positions: Vec<i64>,
    pub saving: f64,
}

pub fn find_opportunities(graph: &Graph, coeffs: CostCoeffs) -> Result<Vec<FusionOpportunity>> {
    let mut opps = Vec::new();
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        if !is_elementwise(n.kind) {
            continue;
        }
        if let Some((chain, side_inputs, side_positions)) = build_fusion_chain(graph, id)? {
            if chain.len() >= 2 {
                let saving = fusion_saving(graph, &chain, coeffs)?;
                if saving > 0.0 {
                    opps.push(FusionOpportunity {
                        nodes: chain,
                        side_inputs,
                        side_positions,
                        saving,
                    });
                }
            }
        }
    }
    opps.sort_by(|a, b| {
        b.saving
            .partial_cmp(&a.saving)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(opps)
}

fn is_elementwise(kind: OpKind) -> bool {
    // 注意：Rsqrt 故意不在此列——它作为融合边界保留为独立 op，
    // 让 lowering 能发专用 rsqrt kernel（0x5f3759df 位 trick），
    // 而非被融进 elementwise 链变成 Fused 节点
    matches!(
        kind,
        OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Relu
            | OpKind::Gelu
            | OpKind::Sigmoid
            | OpKind::Tanh
            | OpKind::Sqrt
            | OpKind::Exp
            | OpKind::Abs
            | OpKind::Log
    )
}

/// reduce 类算子（改变 shape，只能在融合链最前面，作为链头）
fn is_reduce(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax
    )
}

/// 融合链构建结果：节点序列 + side inputs + side input 位置。
/// None 表示无法安全融合（diamond / 自引用）。
type ChainResult = Option<(Vec<base::NodeId>, Vec<ValueId>, Vec<i64>)>;

/// 从 start 往前建链：找 start 的 elementwise/reduce 前驱，递归。
///
/// 链头允许一个 reduce（reduce→elementwise 融合，含 binary elementwise）。
/// reduce 后不再往前扩展（reduce 是 shape 分界点）。
///
/// binary elementwise 的"非链前驱"输入作为 side input 收集。若某 side input
/// 的定义节点已在链中（diamond），放弃整条链（返回 None）以保证正确性。
///
/// 返回 (chain_nodes head→tail, side_inputs head→tail, side_positions head→tail)。
fn build_fusion_chain(graph: &Graph, start: base::NodeId) -> Result<ChainResult> {
    let mut chain = vec![start];
    let mut side_inputs: Vec<ValueId> = Vec::new();
    let mut side_positions: Vec<i64> = Vec::new();
    let mut current = start;
    // 链中节点集合（用于检测 diamond：side input 不能由链中节点产生）
    let mut in_chain: std::collections::HashSet<base::NodeId> = std::collections::HashSet::new();
    in_chain.insert(start);

    loop {
        let n = graph.node(current)?;
        let n_ins = n.inputs();
        // 找链前驱：第一个独占使用的 elementwise/reduce 前驱
        let mut chain_pred: Option<(base::NodeId, ValueId)> = None;
        for &vin in n_ins {
            let v = graph.value(vin)?;
            let def = v.def_node();
            if def == u32::MAX {
                continue;
            }
            if in_chain.contains(&def) {
                continue;
            }
            let pred = graph.node(def)?;
            if (is_elementwise(pred.kind) || is_reduce(pred.kind))
                && is_exclusively_used(graph, vin, current)?
            {
                chain_pred = Some((def, vin));
                break;
            }
        }

        if let Some((pred_id, pred_vin)) = chain_pred {
            // 自引用检测：若 binary 节点的多个 input 都引用链前驱输出
            // （如 add(r, r)），编码无法表示"binary 两输入都取链输出"，放弃融合
            let pred_uses = n_ins.iter().filter(|&&v| v == pred_vin).count();
            if pred_uses > 1 {
                return Ok(None);
            }
            // 收集当前节点的 side inputs（非链前驱输入）
            // side input 的定义节点不能在链中（否则 diamond，放弃）
            let mut node_side_pos: i64 = -1;
            for (idx, &vin) in n_ins.iter().enumerate() {
                if vin == pred_vin {
                    continue;
                }
                let def = graph.value(vin)?.def_node();
                if def != u32::MAX && in_chain.contains(&def) {
                    // diamond：side input 由链中节点产生，无法安全融合
                    return Ok(None);
                }
                side_inputs.push(vin);
                // 记录该 side input 在当前节点 inputs 中的下标
                // （多个 side input 时取首个；binary 最多 1 个 side input）
                if node_side_pos == -1 {
                    node_side_pos = idx as i64;
                }
            }
            side_positions.push(node_side_pos);

            chain.push(pred_id);
            in_chain.insert(pred_id);
            current = pred_id;

            // reduce 作为链头，不再往前
            if is_reduce(graph.node(pred_id)?.kind) {
                break;
            }
        } else {
            // current 无链前驱；它自身的 side_positions 由调用方补
            // （current 是链头时无 side input 概念，记 -1）
            break;
        }
    }

    // 链头（最后一个 current）无 side input
    side_positions.push(-1);

    chain.reverse();
    side_inputs.reverse();
    side_positions.reverse();
    Ok(Some((chain, side_inputs, side_positions)))
}

fn is_exclusively_used(graph: &Graph, v: ValueId, consumer: base::NodeId) -> Result<bool> {
    for id in graph.node_ids() {
        if id == consumer {
            continue;
        }
        let n = graph.node(id)?;
        if n.inputs().contains(&v) {
            return Ok(false);
        }
    }
    if graph.outputs().contains(&v) {
        return Ok(false);
    }
    Ok(true)
}

/// 应用融合：把每条链的链尾节点改成 Fused，inputs 重写为 链头 inputs + side inputs。
/// 链中其余节点变死代码（DCE 清理）。返回应用次数。
///
/// 机会按 saving 降序处理；已被某条融合链消费的节点不再参与后续链（避免重叠
/// 子链被重复改写）。
pub fn apply_fusion(graph: &mut Graph, coeffs: CostCoeffs) -> Result<usize> {
    let opps = find_opportunities(graph, coeffs)?;
    let mut applied = 0usize;
    let mut to_remove: std::collections::HashSet<base::NodeId> = std::collections::HashSet::new();
    // 已被某条融合链消费的节点（含链尾 Fused 节点本身），后续链不可再碰
    let mut consumed: std::collections::HashSet<base::NodeId> = std::collections::HashSet::new();

    for opp in opps {
        let chain = &opp.nodes;
        if chain.len() < 2 {
            continue;
        }
        // 跳过与已应用链重叠的机会
        if chain.iter().any(|&n| consumed.contains(&n)) {
            continue;
        }

        let head = chain[0];
        let tail = chain[chain.len() - 1];

        // 融合后 inputs = 链头 inputs + side inputs（按链序）
        let head_inputs: Vec<ValueId> = graph.node(head)?.inputs().to_vec();
        let mut fused_inputs = head_inputs.clone();
        fused_inputs.extend(opp.side_inputs.iter().copied());

        // op 序列（每个是 OpKind as u8）
        let op_seq: Vec<i64> = chain
            .iter()
            .map(|&n| graph.node(n).map(|v| v.kind as u8 as i64).unwrap_or(0))
            .collect();

        // 把链尾节点改成 Fused，inputs 重写
        graph.storage.set_node_inputs(tail, &fused_inputs);
        graph.storage.node_hdr[tail as usize].op_tag = OpKind::Fused as u8;
        // op 序列 → Shape attr；side input 位置 → Strides attr
        graph
            .storage
            .add_attr_int_array(tail, base::StorageAttrKey::Shape, &op_seq);
        graph
            .storage
            .add_attr_int_array(tail, base::StorageAttrKey::Strides, &opp.side_positions);

        // 若链头是 reduce，复制其 Axis attr 到融合节点（保留 reduce 轴信息）
        let head_kind = graph.node(head)?.kind;
        if is_reduce(head_kind) {
            let mut axis_val: Option<i64> = None;
            for e in graph.node(head)?.attrs() {
                if e.key == base::StorageAttrKey::Axis as u8
                    && e.tag == base::storage::AttrTag::Int as u8
                {
                    axis_val = Some(graph.node(head)?.storage.attr_int(e));
                    break;
                }
            }
            if let Some(ax) = axis_val {
                graph
                    .storage
                    .add_attr_int(tail, base::StorageAttrKey::Axis, ax);
            }
        }

        // 全部链节点标记消费；链中除链尾外标记删除
        for &n in chain {
            consumed.insert(n);
        }
        for &n in &chain[..chain.len() - 1] {
            to_remove.insert(n);
        }
        applied += 1;
    }

    if !to_remove.is_empty() {
        let (new_graph, _, _) = graph.compact(&to_remove);
        *graph = new_graph;
    }
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    #[test]
    fn finds_elementwise_chain() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("x"),
        );
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("r"),
            relu,
        );
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[r_out]);
        let add = g.add_node(OpKind::Add);
        let a_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("a"),
            add,
        );
        g.storage.set_node_inputs(add, &[r_out, x]);
        g.storage.set_node_outputs(add, &[a_out]);
        g.mark_output(a_out);
        let opps = find_opportunities(&g, CostCoeffs::cuda()).unwrap();
        assert!(!opps.is_empty());
    }

    #[test]
    fn apply_fuses_chain_to_custom() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("x"),
        );
        g.mark_input(x);
        // relu(x) -> sigmoid(.) -> 输出
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("r"),
            relu,
        );
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[r_out]);
        let sigmoid = g.add_node(OpKind::Sigmoid);
        let s_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("s"),
            sigmoid,
        );
        g.storage.set_node_inputs(sigmoid, &[r_out]);
        g.storage.set_node_outputs(sigmoid, &[s_out]);
        g.mark_output(s_out);

        let count = apply_fusion(&mut g, CostCoeffs::cuda()).unwrap();
        assert!(count >= 1, "应至少融合一条链");
        // 融合后应只剩 1 个节点（Fused），relu 被删
        assert_eq!(g.node_count(), 1, "融合后应剩 1 个 Fused 节点");
        let n = g.node(0).unwrap();
        assert_eq!(n.kind, OpKind::Fused, "链尾应改成 Fused");
        // 输入应重写为链头输入 x（compact 后 x 仍存在，因它是图输入）
        assert_eq!(n.inputs().len(), 1, "应有 1 个输入");
        assert_eq!(g.inputs().len(), 1, "图应保留 1 个输入 x");
    }

    #[test]
    fn no_fusion_for_single_node() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("x"),
        );
        let relu = g.add_node(OpKind::Relu);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("o"),
            relu,
        );
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[out]);
        g.mark_output(out);
        // 单节点链不应融合
        let count = apply_fusion(&mut g, CostCoeffs::cuda()).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn reduce_then_unary_elementwise_fuses() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4, 8],
            },
            Some("x"),
        );
        g.mark_input(x);
        // ReduceSum(x, axis=1) -> sigmoid(.) -> out
        let rs = g.add_node(OpKind::ReduceSum);
        let r_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("r"),
            rs,
        );
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[r_out]);
        g.storage.add_attr_int(rs, base::StorageAttrKey::Axis, 1);
        let sig = g.add_node(OpKind::Sigmoid);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("o"),
            sig,
        );
        g.storage.set_node_inputs(sig, &[r_out]);
        g.storage.set_node_outputs(sig, &[out]);
        g.mark_output(out);

        let count = apply_fusion(&mut g, CostCoeffs::cuda()).unwrap();
        assert_eq!(count, 1);
        // 融合后应只剩 1 个 Fused 节点（原 sigmoid 尾节点改 Fused，reduce 被 compact 删）
        let customs: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::Fused)
            .collect();
        assert_eq!(customs.len(), 1);
        // Fused 节点应保留 reduce 的 axis attr
        let custom = customs[0];
        let has_axis = g
            .node(custom)
            .unwrap()
            .attrs()
            .iter()
            .any(|e| e.key == base::StorageAttrKey::Axis as u8);
        assert!(has_axis, "融合节点应保留 reduce 的 axis attr");
        // reduce 节点应已被 compact 删除
        let reduces: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::ReduceSum)
            .collect();
        assert!(reduces.is_empty(), "reduce 节点应被融合删除");
    }

    #[test]
    fn reduce_not_fused_when_output_shared() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4, 8],
            },
            Some("x"),
        );
        g.mark_input(x);
        let rs = g.add_node(OpKind::ReduceSum);
        let r_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("r"),
            rs,
        );
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[r_out]);
        g.storage.add_attr_int(rs, base::StorageAttrKey::Axis, 1);
        // sigmoid 用 r_out
        let sig = g.add_node(OpKind::Sigmoid);
        let s_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("s"),
            sig,
        );
        g.storage.set_node_inputs(sig, &[r_out]);
        g.storage.set_node_outputs(sig, &[s_out]);
        // tanh 也用 r_out（非独占）
        let tanh = g.add_node(OpKind::Tanh);
        let t_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("t"),
            tanh,
        );
        g.storage.set_node_inputs(tanh, &[r_out]);
        g.storage.set_node_outputs(tanh, &[t_out]);
        g.mark_output(s_out);
        g.mark_output(t_out);
        // r_out 被两个节点使用，非独占，不应融合
        let count = apply_fusion(&mut g, CostCoeffs::cuda()).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn reduce_then_binary_elementwise_fuses() {
        // ReduceMean(x, axis=1) -> Add(., bias) -> out
        // binary elementwise 接 reduce，bias 是 side input
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4, 8],
            },
            Some("x"),
        );
        g.mark_input(x);
        let bias = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("bias"),
        );
        g.mark_input(bias);
        let rs = g.add_node(OpKind::ReduceMean);
        let r_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("r"),
            rs,
        );
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[r_out]);
        g.storage.add_attr_int(rs, base::StorageAttrKey::Axis, 1);
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("o"),
            add,
        );
        g.storage.set_node_inputs(add, &[r_out, bias]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let count = apply_fusion(&mut g, CostCoeffs::cuda()).unwrap();
        assert_eq!(count, 1, "reduce→add 应融合");
        // 融合后应只剩 1 个 Fused 节点
        let customs: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::Fused)
            .collect();
        assert_eq!(customs.len(), 1);
        let custom = customs[0];
        // inputs 应 = [x, bias]（链头 reduce 的 input x + side input bias）
        let ins = g.node(custom).unwrap().inputs();
        assert_eq!(ins.len(), 2, "融合节点应有 2 个输入 [x, bias]");
        // 验证两个输入都能取到（不是 u32::MAX）
        assert_ne!(ins[0], u32::MAX, "x 应被保留");
        assert_ne!(ins[1], u32::MAX, "bias 应作为 side input 保留");
        // 两个输入应不同（x 和 bias）
        assert_ne!(ins[0], ins[1], "x 与 bias 是不同 value");
        // 应保留 reduce 的 axis attr
        let has_axis = g
            .node(custom)
            .unwrap()
            .attrs()
            .iter()
            .any(|e| e.key == base::StorageAttrKey::Axis as u8);
        assert!(has_axis, "应保留 reduce 的 axis attr");
    }

    #[test]
    fn binary_chain_preserves_side_inputs() {
        // relu(x) -> add(., b) -> mul(., c) -> out
        // 两个 binary 各有 side input，融合后 inputs = [x, b, c]
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("x"),
        );
        g.mark_input(x);
        let b = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("b"),
        );
        g.mark_input(b);
        let c = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("c"),
        );
        g.mark_input(c);
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("r"),
            relu,
        );
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[r_out]);
        let add = g.add_node(OpKind::Add);
        let a_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("a"),
            add,
        );
        g.storage.set_node_inputs(add, &[r_out, b]);
        g.storage.set_node_outputs(add, &[a_out]);
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("o"),
            mul,
        );
        g.storage.set_node_inputs(mul, &[a_out, c]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);

        let count = apply_fusion(&mut g, CostCoeffs::cuda()).unwrap();
        assert_eq!(count, 1, "relu→add→mul 应融成一条链");
        let customs: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::Fused)
            .collect();
        assert_eq!(customs.len(), 1);
        let ins = g.node(customs[0]).unwrap().inputs();
        assert_eq!(ins.len(), 3, "融合节点 inputs 应为 [x, b, c]");
        // 三个输入应分别对应 x, b, c（都不是 MAX 且互不相同）
        assert_ne!(ins[0], u32::MAX);
        assert_ne!(ins[1], u32::MAX);
        assert_ne!(ins[2], u32::MAX);
        // 验证 Strides attr 记录了 side input 位置
        let strides: Option<Vec<i64>> = {
            let n = g.node(customs[0]).unwrap();
            let mut found = None;
            for e in n.attrs() {
                if e.key == base::StorageAttrKey::Strides as u8 {
                    found = Some(n.storage.attr_int_array(e).to_vec());
                    break;
                }
            }
            found
        };
        let strides = strides.expect("应有 Strides attr");
        // 链 head→tail: [relu, add, mul]，side_positions: [-1, 1, 1]
        // relu 无 side input(-1)，add 的 side input b 在 inputs[1]，mul 的 side input c 在 inputs[1]
        assert_eq!(strides, vec![-1, 1, 1], "side input 位置应为 [-1, 1, 1]");
    }

    #[test]
    fn diamond_side_input_not_fused() {
        // relu(x) -> add(r_out, r_out)：add 的两个输入都是同一个链前驱输出（自引用）
        // 编码无法表示"binary 两输入都取链输出"，应放弃融合
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("x"),
        );
        g.mark_input(x);
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("r"),
            relu,
        );
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[r_out]);
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4],
            },
            Some("o"),
            add,
        );
        // 两个输入都是 r_out（自引用）
        g.storage.set_node_inputs(add, &[r_out, r_out]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let count = apply_fusion(&mut g, CostCoeffs::cuda()).unwrap();
        assert_eq!(count, 0, "自引用 binary（两输入都取链输出）不应融合");
    }
}
