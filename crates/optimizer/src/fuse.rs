//! fuse — 多对一启发式融合（基于 cost model）
//!
//! 设计哲学：把可融合的算子链合成单个 Custom 节点，省 launch overhead +
//! 中间结果访存。用 cost_model 判定融合收益（只融合收益 > 0 的链）。
//!
//! 融合策略：
//! - elementwise 链（Add/Relu/Sigmoid 等单输入单输出算子链）
//! - 链尾节点 op 改成 Custom，inputs 重写为链头 inputs，attr 记录原始 op 序列
//! - 链中其余节点变死代码，由 DCE 清理

use crate::cost_model::{fusion_saving, CostCoeffs};
use base::{Graph, OpKind, Result, ValueId};

pub struct FusionOpportunity {
    pub nodes: Vec<base::NodeId>,
    pub saving: f64,
}

pub fn find_opportunities(graph: &Graph, coeffs: CostCoeffs) -> Result<Vec<FusionOpportunity>> {
    let mut opps = Vec::new();
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        if !is_elementwise(n.kind) {
            continue;
        }
        let chain = build_fusion_chain(graph, id)?;
        if chain.len() >= 2 {
            let saving = fusion_saving(graph, &chain, coeffs)?;
            if saving > 0.0 {
                opps.push(FusionOpportunity {
                    nodes: chain,
                    saving,
                });
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
    )
}

/// 从 start 往前建链：找 start 的 elementwise 前驱，递归
fn build_fusion_chain(graph: &Graph, start: base::NodeId) -> Result<Vec<base::NodeId>> {
    let mut chain = vec![start];
    let mut current = start;
    loop {
        let n = graph.node(current)?;
        let mut next = None;
        for &vin in n.inputs() {
            let v = graph.value(vin)?;
            let def = v.def_node();
            if def == u32::MAX {
                continue;
            }
            let pred = graph.node(def)?;
            if is_elementwise(pred.kind) && is_exclusively_used(graph, vin, current)? {
                next = Some(def);
                break;
            }
        }
        match next {
            Some(pred) => {
                chain.push(pred);
                current = pred;
            }
            None => break,
        }
    }
    chain.reverse();
    Ok(chain)
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

/// 应用融合：把每条链的链尾节点改成 Custom，inputs 重写为链头 inputs。
/// 链中其余节点变死代码（DCE 清理）。返回应用次数。
pub fn apply_fusion(graph: &mut Graph, coeffs: CostCoeffs) -> Result<usize> {
    let opps = find_opportunities(graph, coeffs)?;
    let mut applied = 0usize;
    let mut to_remove: std::collections::HashSet<base::NodeId> = std::collections::HashSet::new();

    for opp in opps {
        let chain = &opp.nodes;
        if chain.len() < 2 {
            continue;
        }
        let head = chain[0];
        let tail = chain[chain.len() - 1];

        // 链头的 inputs 作为融合后节点的 inputs
        let head_inputs: Vec<ValueId> = graph.node(head)?.inputs().to_vec();
        // 链尾的 op 序列（用于 attr 记录）
        let op_seq: Vec<i64> = chain
            .iter()
            .map(|&n| graph.node(n).map(|v| v.kind as u8 as i64).unwrap_or(0))
            .collect();

        // 把链尾节点改成 Custom，inputs 重写
        graph.storage.set_node_inputs(tail, &head_inputs);
        graph.storage.node_hdr[tail as usize].op_tag = OpKind::Custom as u8;
        // 记录融合的 op 序列到 attr（复用 Shape 的 int array 槽位）
        graph
            .storage
            .add_attr_int_array(tail, base::StorageAttrKey::Shape, &op_seq);

        // 链中其余节点（除链尾）标记删除
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
        // 融合后应只剩 1 个节点（Custom），relu 被删
        assert_eq!(g.node_count(), 1, "融合后应剩 1 个 Custom 节点");
        let n = g.node(0).unwrap();
        assert_eq!(n.kind, OpKind::Custom, "链尾应改成 Custom");
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
}
