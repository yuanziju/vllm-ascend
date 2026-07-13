//! extract — 从 IR Graph 提取 KernelSpec 列表
//!
//! 遍历图中每个计算节点，收集其输入/输出张量的 shape+dtype，
//! 以及算子属性（axis/perm/shape/epsilon/value 等），
//! 打包成 KernelSpec 供后端代码生成器使用。

use crate::spec::*;
use base::storage::AttrTag;
use base::{Graph, NodeView, OpKind};

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
    let dtype = outputs.first().map(|t| t.dtype).unwrap_or(base::DType::F32);

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

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, Type};

    /// 构造简单图：x → add(x,x) → out，验证 extract_kernels 能提取出
    /// 节点的 KernelSpec（name、inputs、outputs、dtype 都正确）。
    #[test]
    fn extract_kernels_basic() {
        let mut g = Graph::new("test");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![2, 3],
        };
        let x = g.add_input(ty.clone(), Some("x"));
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(ty, Some("out"), add);
        g.storage.set_node_inputs(add, &[x, x]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let specs = extract_kernels(&g);
        assert_eq!(specs.len(), 1, "应有 1 个 KernelSpec");
        let spec = &specs[0];
        assert_eq!(spec.op, OpKind::Add);
        assert_eq!(spec.inputs.len(), 2, "Add 应有 2 个输入");
        assert_eq!(spec.outputs.len(), 1, "Add 应有 1 个输出");
        assert_eq!(spec.dtype, DType::F32);
        assert!(spec.name.starts_with("neutron_add_"), "name 应为 neutron_add_*, 实际: {}", spec.name);
    }

    /// 空图应返回空 KernelSpec 列表，不 panic
    #[test]
    fn extract_kernels_empty_graph() {
        let g = Graph::new("empty");
        let specs = extract_kernels(&g);
        assert!(specs.is_empty(), "空图应返回空 KernelSpec 列表");
    }

    /// 多节点图：每个计算节点都应有一个 KernelSpec
    #[test]
    fn extract_kernels_multi_node() {
        let mut g = Graph::new("multi");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![4, 8],
        };
        let x = g.add_input(ty.clone(), Some("x"));
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(ty.clone(), Some("r"), relu);
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[r_out]);
        let sqrt = g.add_node(OpKind::Sqrt);
        let out = g.add_value(ty, Some("out"), sqrt);
        g.storage.set_node_inputs(sqrt, &[r_out]);
        g.storage.set_node_outputs(sqrt, &[out]);
        g.mark_output(out);

        let specs = extract_kernels(&g);
        assert_eq!(specs.len(), 2, "应有 2 个 KernelSpec (relu + sqrt)");
        assert_eq!(specs[0].op, OpKind::Relu);
        assert_eq!(specs[1].op, OpKind::Sqrt);
        // 名字应按节点顺序编号
        assert!(specs[0].name.ends_with("_0"), "第一个 kernel 名字应以 _0 结尾: {}", specs[0].name);
        assert!(specs[1].name.ends_with("_1"), "第二个 kernel 名字应以 _1 结尾: {}", specs[1].name);
    }

    /// TensorSpec 应正确提取 shape 和 name
    #[test]
    fn extract_tensor_metadata() {
        let mut g = Graph::new("meta");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![16, 32],
        };
        let x = g.add_input(ty.clone(), Some("input_x"));
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(ty, Some("output_y"), add);
        g.storage.set_node_inputs(add, &[x, x]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let specs = extract_kernels(&g);
        let spec = &specs[0];
        // 输入应有 name="input_x"，shape=[16, 32]
        let in0 = &spec.inputs[0];
        assert_eq!(in0.name, "input_x");
        assert_eq!(in0.dims, vec![16, 32]);
        assert!(in0.is_input);
        // 输出应有 name="output_y"
        let out0 = &spec.outputs[0];
        assert_eq!(out0.name, "output_y");
        assert_eq!(out0.dims, vec![16, 32]);
        assert!(!out0.is_input);
    }

    /// op_short_name 每个变体都要返回非空短名
    #[test]
    fn op_short_name_all_variants_nonempty() {
        // 添加新 OpKind 变体时 op_short_name 漏写分支会编译失败
        // (non-exhaustive match)，但仍要确保返回值非空
        for op in [
            OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div,
            OpKind::MatMul, OpKind::Relu, OpKind::Gelu, OpKind::Sigmoid,
            OpKind::Tanh, OpKind::Softmax, OpKind::LayerNorm, OpKind::Conv,
            OpKind::Pool, OpKind::Reshape, OpKind::Transpose, OpKind::Concat,
            OpKind::Slice, OpKind::Constant, OpKind::Placeholder, OpKind::Return,
            OpKind::Sqrt, OpKind::Exp, OpKind::Pow,
            OpKind::ReduceSum, OpKind::ReduceMean, OpKind::ReduceMax,
            OpKind::Rsqrt, OpKind::Reciprocal, OpKind::Abs, OpKind::Log,
            OpKind::Fused, OpKind::Custom,
        ] {
            let name = op_short_name(op);
            assert!(!name.is_empty(), "OpKind::{:?} short name 为空", op);
        }
    }
}
