//! lowering — 架构无关图 → 目标架构图

use crate::{ArchGraph, ArchOp};
use base::{Graph, OpKind, Result};

pub fn lower(graph: &Graph, _target: common::Target) -> Result<ArchGraph> {
    let mut ag = ArchGraph::new(_target);
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        let native = match n.kind {
            OpKind::MatMul => "mma",
            OpKind::Add => "add",
            OpKind::Sub => "sub",
            OpKind::Mul => "mul",
            OpKind::Div => "div",
            OpKind::Relu => "relu",
            OpKind::Gelu => "gelu",
            OpKind::Sigmoid => "sigmoid",
            OpKind::Tanh => "tanh",
            OpKind::Softmax => "softmax",
            OpKind::LayerNorm => "layer_norm",
            OpKind::Conv => "conv",
            OpKind::Sqrt => "sqrt",
            OpKind::Rsqrt => "rsqrt",
            OpKind::Exp => "exp",
            OpKind::Pow => "pow",
            OpKind::ReduceSum => "reduce_sum",
            OpKind::ReduceMean => "reduce_mean",
            OpKind::ReduceMax => "reduce_max",
            OpKind::Constant => "const",
            OpKind::Placeholder => "load",
            OpKind::Return => "store",
            // Fused：融合产物，发 "fused" kernel（attr 记 op 序列供后端重建）
            OpKind::Fused => "fused",
            // Custom：未知 ONNX 算子（attr 记原始 op_type 字符码），透传 op 名
            OpKind::Custom => "custom",
            other => {
                return Err(base::NeutronError::Backend(format!(
                    "lowering 未覆盖: {:?}",
                    other
                )))
            }
        };
        ag.add(ArchOp::KernelCall(native.to_string()));
    }
    Ok(ag)
}
