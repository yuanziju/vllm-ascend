//! lowering — 架构无关图 → 目标架构图

use base::{Graph, OpKind, Result};
use crate::{ArchGraph, ArchOp};

pub fn lower(graph: &Graph, _target: common::Target) -> Result<ArchGraph> {
    let mut ag = ArchGraph::new(_target);
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        let native = match n.kind {
            OpKind::MatMul => "mma",
            OpKind::Add => "add",
            OpKind::Relu => "relu",
            OpKind::Conv => "conv",
            OpKind::Softmax => "softmax",
            OpKind::LayerNorm => "layer_norm",
            OpKind::Placeholder => "load",
            OpKind::Return => "store",
            other => return Err(base::NeutronError::Backend(format!(
                "lowering 未覆盖: {:?}",
                other
            ))),
        };
        ag.add(ArchOp::KernelCall(native.to_string()));
    }
    Ok(ag)
}
