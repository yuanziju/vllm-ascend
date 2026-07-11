//! extract — 从 IR Graph 提取 KernelSpec 列表
//!
//! 遍历图中每个计算节点，收集其输入/输出张量的 shape+dtype，
//! 以及算子属性（axis/perm/shape/epsilon/value 等），
//! 打包成 KernelSpec 供后端代码生成器使用。

use crate::spec::*;
use base::{Graph, NodeView, OpKind};
use base::storage::AttrTag;

/// 从 IR Graph 提取所有计算节点的 KernelSpec
///
/// Constant/Placeholder/Return 节点也提取（后端可能需要生成数据移动代码）。
pub fn extract_kernels(graph: &Graph) -> Vec<KernelSpec> {
    let mut specs = Vec::new();
    for (idx, node_id) in graph.node_ids().enumerate() {
        let node = match graph.node(node_id) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let spec = extract_one(graph, &node, idx as u32);
        specs.push(spec);
    }
    specs
}

fn extract_one(graph: &Graph, node: &NodeView, idx: u32) -> KernelSpec {
    let inputs = node
        .inputs()
        .iter()
        .map(|&vid| extract_tensor(graph, vid, true))
        .collect();
    let outputs: Vec<TensorSpec> = node
        .outputs()
        .iter()
        .map(|&vid| extract_tensor(graph, vid, false))
        .collect();
    let attrs = extract_attrs(node);
    let dtype = outputs
        .first()
        .map(|t| t.dtype)
        .unwrap_or(base::DType::F32)
        .clone();

    KernelSpec {
        name: format!("neutron_{}_{}", op_short_name(node.kind), idx),
        op: node.kind,
        inputs,
        outputs,
        attrs,
        dtype,
        node_idx: idx,
    }
}

fn extract_tensor(graph: &Graph, vid: base::ValueId, is_input: bool) -> TensorSpec {
    let hdr = &graph.storage.value_hdr[vid as usize];
    let type_tag = base::TypeTag::from_u8(hdr.type_tag).unwrap_or(base::TypeTag::ScalarF32);
    let dtype = type_tag.dtype();
    let dims = graph.storage.value_shape(vid).to_vec();
    let name = graph
        .storage
        .value_name(vid)
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("t{}", vid));
    TensorSpec {
        name,
        dims,
        dtype,
        is_input,
    }
}

fn extract_attrs(node: &NodeView) -> KernelAttrs {
    // AttrKey 数值常量（base::storage::AttrKey 是私有的，这里用数值匹配）
    const KEY_STRIDES: u8 = 0;
    const KEY_PADDING: u8 = 1;
    const KEY_DILATION: u8 = 2;
    const KEY_GROUPS: u8 = 3;
    const KEY_AXIS: u8 = 4;
    const KEY_EPSILON: u8 = 9;
    const KEY_SHAPE: u8 = 10;
    const KEY_VALUE: u8 = 11;
    const KEY_PERM: u8 = 12;

    let mut attrs = KernelAttrs::default();
    for e in node.attrs() {
        match (e.key, e.tag) {
            (k, t) if k == KEY_AXIS && t == AttrTag::Int as u8 => {
                attrs.axis = Some(node.storage.attr_int(e));
            }
            (k, t) if k == KEY_EPSILON && t == AttrTag::Float as u8 => {
                attrs.epsilon = Some(node.storage.attr_float(e));
            }
            (k, t) if k == KEY_VALUE && t == AttrTag::Float as u8 => {
                attrs.value = Some(node.storage.attr_float(e));
            }
            (k, t) if k == KEY_VALUE && t == AttrTag::FloatArray as u8 => {
                attrs.tensor_data = node.storage.attr_float_array(e).to_vec();
            }
            (k, t) if k == KEY_PERM && t == AttrTag::IntArray as u8 => {
                attrs.perm = node.storage.attr_int_array(e).to_vec();
            }
            (k, t) if k == KEY_SHAPE && t == AttrTag::IntArray as u8 => {
                attrs.shape = node.storage.attr_int_array(e).to_vec();
            }
            (k, t) if k == KEY_STRIDES && t == AttrTag::IntArray as u8 => {
                attrs.strides = node.storage.attr_int_array(e).to_vec();
                attrs.conv_stride = attrs.strides.clone();
                attrs.pool_stride = attrs.strides.clone();
            }
            (k, t) if k == KEY_GROUPS && t == AttrTag::Int as u8 => {
                attrs.conv_groups = Some(node.storage.attr_int(e));
            }
            (k, t) if k == KEY_PADDING && t == AttrTag::IntArray as u8 => {
                attrs.conv_padding = node.storage.attr_int_array(e).to_vec();
                attrs.pool_padding = attrs.conv_padding.clone();
            }
            (k, t) if k == KEY_DILATION && t == AttrTag::IntArray as u8 => {
                attrs.conv_dilation = node.storage.attr_int_array(e).to_vec();
            }
            _ => {}
        }
    }

    if node.kind == OpKind::Custom {
        attrs.custom_op_type = "unknown".to_string();
    }

    attrs
}

fn op_short_name(op: OpKind) -> &'static str {
    match op {
        OpKind::Add => "add",
        OpKind::Sub => "sub",
        OpKind::Mul => "mul",
        OpKind::Div => "div",
        OpKind::MatMul => "matmul",
        OpKind::Relu => "relu",
        OpKind::Gelu => "gelu",
        OpKind::Sigmoid => "sigmoid",
        OpKind::Tanh => "tanh",
        OpKind::Softmax => "softmax",
        OpKind::LayerNorm => "layernorm",
        OpKind::Conv => "conv",
        OpKind::Pool => "pool",
        OpKind::Reshape => "reshape",
        OpKind::Transpose => "transpose",
        OpKind::Concat => "concat",
        OpKind::Slice => "slice",
        OpKind::Constant => "const",
        OpKind::Placeholder => "placeholder",
        OpKind::Return => "ret",
        OpKind::Sqrt => "sqrt",
        OpKind::Exp => "exp",
        OpKind::Pow => "pow",
        OpKind::ReduceSum => "reduce_sum",
        OpKind::ReduceMean => "reduce_mean",
        OpKind::ReduceMax => "reduce_max",
        OpKind::Rsqrt => "rsqrt",
        OpKind::Reciprocal => "reciprocal",
        OpKind::Abs => "abs",
        OpKind::Log => "log",
        OpKind::Fused => "fused",
        OpKind::Custom => "custom",
    }
}
