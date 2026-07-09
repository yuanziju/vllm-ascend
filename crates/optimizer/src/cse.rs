//! cse — 公共子表达式消除（IO 同样性）

use base::{Graph, NodeView, OpKind, Result, ValueId};
use std::collections::HashMap;

pub struct CseResult {
    pub removed_nodes: Vec<base::NodeId>,
    pub value_replacements: HashMap<ValueId, ValueId>,
}

pub fn run_cse(graph: &Graph) -> Result<CseResult> {
    let mut signatures: HashMap<OpSignature, base::NodeId> = HashMap::new();
    let mut value_replacements: Vec<(ValueId, ValueId)> = Vec::new();
    let mut removed_nodes: Vec<base::NodeId> = Vec::new();

    for id in graph.node_ids() {
        let n = graph.node(id)?;
        let sig = OpSignature {
            kind: n.kind,
            inputs: n.inputs().to_vec(),
        };
        if let Some(&existing) = signatures.get(&sig) {
            let existing_node = graph.node(existing)?;
            let existing_outputs = existing_node.outputs();
            let current_outputs = n.outputs();
            for (old, new) in current_outputs.iter().zip(existing_outputs.iter()) {
                value_replacements.push((*old, *new));
            }
            removed_nodes.push(id);
        } else {
            signatures.insert(sig, id);
        }
    }

    let value_replacements: HashMap<ValueId, ValueId> = value_replacements.into_iter().collect();
    Ok(CseResult {
        removed_nodes,
        value_replacements,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OpSignature {
    kind: OpKind,
    inputs: Vec<ValueId>,
}

pub fn apply_cse(graph: &mut Graph) -> Result<usize> {
    let result = run_cse(graph)?;
    let removed_count = result.removed_nodes.len();
    if removed_count == 0 {
        return Ok(0);
    }
    let remove_set: std::collections::HashSet<base::NodeId> =
        result.removed_nodes.into_iter().collect();
    let (new_graph, _, _) = graph.compact(&remove_set);
    *graph = new_graph;
    Ok(removed_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    #[test]
    fn cse_finds_redundant_add() {
        let mut g = Graph::new("test");
        let a = g.add_input(
            Type::Tensor { dtype: DType::F32, dims: vec![2, 2] },
            Some("a"),
        );
        let b = g.add_input(
            Type::Tensor { dtype: DType::F32, dims: vec![2, 2] },
            Some("b"),
        );
        let add1 = g.add_node(OpKind::Add);
        let out1 = g.add_value(
            Type::Tensor { dtype: DType::F32, dims: vec![2, 2] },
            Some("c"),
            add1,
        );
        g.raw.set_node_inputs(add1, &[a, b]);
        g.raw.set_node_outputs(add1, &[out1]);
        let add2 = g.add_node(OpKind::Add);
        let out2 = g.add_value(
            Type::Tensor { dtype: DType::F32, dims: vec![2, 2] },
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
}
