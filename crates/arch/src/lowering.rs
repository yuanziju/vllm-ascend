//! lowering — 架构无关图 → 目标架构图

use crate::{ArchGraph, ArchOp};
use base::{Graph, OpKind, Result, ValueId};

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
            OpKind::Reciprocal => "reciprocal",
            // Abs/Log：单输入，直发同名 kernel
            OpKind::Abs => "abs",
            OpKind::Log => "log",
            OpKind::Exp => "exp",
            OpKind::Pow => "pow",
            OpKind::ReduceSum => "reduce_sum",
            OpKind::ReduceMean => "reduce_mean",
            OpKind::ReduceMax => "reduce_max",
            // 数据移动（无 FLOPs，仅布局/形状调整）
            OpKind::Reshape => "reshape",
            OpKind::Transpose => "transpose",
            OpKind::Concat => "concat",
            OpKind::Slice => "slice",
            OpKind::Pool => "pool",
            OpKind::Constant => "const",
            OpKind::Placeholder => "load",
            OpKind::Return => "store",
            // Fused：融合产物，发 "fused" kernel（attr 记 op 序列供后端重建）
            OpKind::Fused => "fused",
            // Custom：未知 ONNX 算子（attr 记原始 op_type 字符码），透传 op 名
            OpKind::Custom => "custom",
            // 所有 OpKind 变体已显式覆盖；新增 op 时编译器会因 non-exhaustive 报错，
            // 强制在此补 lowering 分支——比 catch-all 更安全（不会静默漏）
        };
        // 透传 IR 节点的 inputs/outputs（ValueId），让寄存器分配能读 def-use
        let inputs: Vec<ValueId> = n.inputs().to_vec();
        let outputs: Vec<ValueId> = n.outputs().to_vec();
        ag.add(ArchOp::KernelCall {
            name: native.to_string(),
            inputs,
            outputs,
        });
    }
    Ok(ag)
}
