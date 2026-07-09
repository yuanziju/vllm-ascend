//! fuse — 多对一启发式融合（基于 cost model）

use base::{Graph, NodeView, OpKind, Result, ValueId};
use crate::cost_model::{fusion_saving, CostCoeffs};

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
                opps.push(FusionOpportunity { nodes: chain, saving });
            }
        }
    }
    opps.sort_by(|a, b| b.saving.partial_cmp(&a.saving).unwrap_or(std::cmp::Ordering::Equal));
    Ok(opps)
}

fn is_elementwise(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div
            | OpKind::Relu | OpKind::Gelu | OpKind::Sigmoid | OpKind::Tanh
    )
}

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

pub fn apply_fusion(graph: &mut Graph, coeffs: CostCoeffs) -> Result<usize> {
    let opps = find_opportunities(graph, coeffs)?;
    Ok(opps.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    #[test]
    fn finds_elementwise_chain() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor { dtype: DType::F32, dims: vec![4] },
            Some("x"),
        );
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(
            Type::Tensor { dtype: DType::F32, dims: vec![4] },
            Some("r"),
            relu,
        );
        g.raw.set_node_inputs(relu, &[x]);
        g.raw.set_node_outputs(relu, &[r_out]);
        let add = g.add_node(OpKind::Add);
        let a_out = g.add_value(
            Type::Tensor { dtype: DType::F32, dims: vec![4] },
            Some("a"),
            add,
        );
        g.raw.set_node_inputs(add, &[r_out, x]);
        g.raw.set_node_outputs(add, &[a_out]);
        g.mark_output(a_out);
        let opps = find_opportunities(&g, CostCoeffs::cuda()).unwrap();
        assert!(!opps.is_empty());
    }
}
