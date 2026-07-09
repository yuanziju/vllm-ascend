//! decompose — 一对多拆分（LayerNorm/Softmax/Gelu → 细粒度原语）

use base::{Graph, OpKind, Result};

pub struct DecomposeResult {
    pub original: base::NodeId,
    pub expanded: Vec<base::NodeId>,
}

pub fn run_decompose(graph: &mut Graph) -> Result<Vec<DecomposeResult>> {
    let mut results = Vec::new();
    let to_decompose: Vec<(base::NodeId, OpKind)> = graph
        .node_ids()
        .filter_map(|id| {
            graph.node(id).ok().and_then(|n| {
                if matches!(n.kind, OpKind::LayerNorm | OpKind::Softmax | OpKind::Gelu) {
                    Some((id, n.kind))
                } else {
                    None
                }
            })
        })
        .collect();

    for (id, kind) in to_decompose {
        let expanded = match kind {
            OpKind::LayerNorm => decompose_layernorm(graph, id)?,
            OpKind::Softmax => decompose_softmax(graph, id)?,
            OpKind::Gelu => decompose_gelu(graph, id)?,
            _ => Vec::new(),
        };
        results.push(DecomposeResult {
            original: id,
            expanded,
        });
    }
    Ok(results)
}

fn decompose_layernorm(_graph: &mut Graph, _id: base::NodeId) -> Result<Vec<base::NodeId>> {
    // TODO: 完整实现
    Ok(Vec::new())
}

fn decompose_softmax(_graph: &mut Graph, _id: base::NodeId) -> Result<Vec<base::NodeId>> {
    // TODO: 完整实现
    Ok(Vec::new())
}

fn decompose_gelu(_graph: &mut Graph, _id: base::NodeId) -> Result<Vec<base::NodeId>> {
    // TODO: 完整实现
    Ok(Vec::new())
}
