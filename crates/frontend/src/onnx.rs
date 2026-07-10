//! onnx — ONNX 解析（手写 protobuf wire-format，无 prost 依赖）
//!
//! 设计哲学：不引入 prost/prost-build（重依赖 + protoc 代码生成），手写
//! 一个只够用的 protobuf 解码器（见 [`proto`]），解出 ONNX ModelProto →
//! GraphProto → NodeProto 的关键字段（op_type / input / output / name），
//! 构建 Neutron 架构无关图。
//!
//! ONNX op_type → Neutron OpKind 映射覆盖常见算子。未知算子映射成 Custom
//! （attr 记录原始 op_type），不报错，保证前向兼容。
//!
//! Protobuf 字段编号参考 ONNX schema：
//! - ModelProto: 7=graph(LEN)
//! - GraphProto: 1=node(repeated NodeProto), 2=name(string), 5=initializer(repeated)
//! - NodeProto: 1=input(repeated string), 2=output(repeated string), 3=op_type(string),
//!   4=name(string), 5=attribute(repeated), 7=domain(string)

use base::StorageAttrKey;
use base::{DType, Graph, OpKind, Result, Type};

use crate::proto::{read_string_field, Cursor};

/// 解析 ONNX 字节流为架构无关图
pub fn parse(bytes: &[u8]) -> Result<Graph> {
    if bytes.is_empty() {
        // 空输入：返回空图（占位节点已无意义，下游 DCE 会处理）
        return Ok(Graph::new("onnx"));
    }

    let model = parse_model(bytes)?;
    let mut g = Graph::new(model.graph_name.as_deref().unwrap_or("onnx"));

    // 第一遍：收集所有 value 名 → ValueId（输入、initializer、节点输出）
    // 用名称注册表做 SSA 去重
    let mut registry = NameRegistry::new();

    // 图输入（GraphProto.input，field 11）作为 graph inputs
    for name in &model.inputs {
        let v = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![-1, -1],
            },
            Some(name),
        );
        registry.register(name.clone(), v);
        g.mark_input(v);
    }
    // initializer（GraphProto.initializer，field 5）作为常量输入
    for name in &model.initializers {
        if registry.get(name).is_none() {
            let v = g.add_input(
                Type::Tensor {
                    dtype: DType::F32,
                    dims: vec![-1],
                },
                Some(name),
            );
            registry.register(name.clone(), v);
        }
    }

    // 节点
    for node in &model.nodes {
        let kind = map_op_type(&node.op_type);
        let nid = g.add_node(kind);
        // 未知 op_type 记录到 attr（Custom 槽位用 Shape int array 存字符 code 点）
        if matches!(kind, OpKind::Custom) {
            let codes: Vec<i64> = node.op_type.chars().map(|c| c as i64).collect();
            g.storage
                .add_attr_int_array(nid, StorageAttrKey::Shape, &codes);
        }
        // outputs：先创建 value（这样 inputs 可以前向引用后续节点输出）
        for out_name in &node.outputs {
            let v = g.add_value(
                Type::Tensor {
                    dtype: DType::F32,
                    dims: vec![-1, -1],
                },
                Some(out_name),
                nid,
            );
            registry.register(out_name.clone(), v);
        }
        g.storage.set_node_outputs(
            nid,
            &node
                .outputs
                .iter()
                .map(|n| registry.get(n).unwrap_or(u32::MAX))
                .collect::<Vec<_>>(),
        );
    }

    // 第二遍：填充每个节点的 inputs（引用已注册的 value）
    for (node_idx, node) in model.nodes.iter().enumerate() {
        let nid = node_idx as u32;
        let inputs: Vec<u32> = node
            .inputs
            .iter()
            .map(|n| registry.get(n).unwrap_or(u32::MAX))
            .collect();
        g.storage.set_node_inputs(nid, &inputs);
    }

    // 图输出（GraphProto.output，field 12）
    for name in &model.outputs {
        if let Some(v) = registry.get(name) {
            g.mark_output(v);
        }
    }

    Ok(g)
}

// --- ONNX 消息结构（解析结果） ---

#[derive(Debug, Default)]
struct ModelInfo {
    graph_name: Option<String>,
    nodes: Vec<NodeInfo>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializers: Vec<String>,
}

#[derive(Debug, Default)]
struct NodeInfo {
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

fn parse_model(bytes: &[u8]) -> Result<ModelInfo> {
    let mut c = Cursor::new(bytes);
    let mut info = ModelInfo::default();
    while !c.eof() {
        let (field, wt) = c.read_tag()?;
        match (field, wt) {
            // ir_version, producer_name 等都跳过
            // ModelProto.graph = field 7, LEN
            (7, 2) => {
                let graph_buf = c.read_length_delimited()?;
                let graph = parse_graph(graph_buf)?;
                info.graph_name = graph.graph_name;
                info.nodes = graph.nodes;
                info.inputs = graph.inputs;
                info.outputs = graph.outputs;
                info.initializers = graph.initializers;
            }
            _ => c.skip_field(wt)?,
        }
    }
    Ok(info)
}

#[derive(Debug, Default)]
struct GraphInfo {
    graph_name: Option<String>,
    nodes: Vec<NodeInfo>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializers: Vec<String>,
}

fn parse_graph(buf: &[u8]) -> Result<GraphInfo> {
    let mut c = Cursor::new(buf);
    let mut g = GraphInfo::default();
    while !c.eof() {
        let (field, wt) = c.read_tag()?;
        match (field, wt) {
            // GraphProto.node = field 1, repeated LEN
            (1, 2) => {
                let node_buf = c.read_length_delimited()?;
                g.nodes.push(parse_node(node_buf)?);
            }
            // GraphProto.name = field 2, string
            (2, 2) => {
                let s = c.read_length_delimited()?;
                g.graph_name = Some(read_string_field(s)?);
            }
            // GraphProto.initializer = field 5, repeated LEN (TensorProto)
            (5, 2) => {
                let tensor_buf = c.read_length_delimited()?;
                if let Some(name) = parse_tensor_name(tensor_buf)? {
                    g.initializers.push(name);
                }
            }
            // GraphProto.input = field 11, repeated ValueInfoProto
            (11, 2) => {
                let vi_buf = c.read_length_delimited()?;
                if let Some(name) = parse_value_info_name(vi_buf)? {
                    g.inputs.push(name);
                }
            }
            // GraphProto.output = field 12, repeated ValueInfoProto
            (12, 2) => {
                let vi_buf = c.read_length_delimited()?;
                if let Some(name) = parse_value_info_name(vi_buf)? {
                    g.outputs.push(name);
                }
            }
            _ => c.skip_field(wt)?,
        }
    }
    Ok(g)
}

fn parse_node(buf: &[u8]) -> Result<NodeInfo> {
    let mut c = Cursor::new(buf);
    let mut n = NodeInfo::default();
    while !c.eof() {
        let (field, wt) = c.read_tag()?;
        match (field, wt) {
            // NodeProto.input = field 1, repeated string
            (1, 2) => {
                let s = c.read_length_delimited()?;
                n.inputs.push(read_string_field(s)?);
            }
            // NodeProto.output = field 2, repeated string
            (2, 2) => {
                let s = c.read_length_delimited()?;
                n.outputs.push(read_string_field(s)?);
            }
            // NodeProto.op_type = field 3, string
            (3, 2) => {
                let s = c.read_length_delimited()?;
                n.op_type = read_string_field(s)?;
            }
            // NodeProto.name = field 4, string（跳过，用 op_type）
            // NodeProto.attribute = field 5（跳过，不解析具体属性）
            // NodeProto.domain = field 7（跳过）
            _ => c.skip_field(wt)?,
        }
    }
    Ok(n)
}

/// 从 TensorProto 解出 name（field 8）。其余字段忽略。
fn parse_tensor_name(buf: &[u8]) -> Result<Option<String>> {
    let mut c = Cursor::new(buf);
    let mut name = None;
    while !c.eof() {
        let (field, wt) = c.read_tag()?;
        match (field, wt) {
            (8, 2) => {
                let s = c.read_length_delimited()?;
                name = Some(read_string_field(s)?);
            }
            _ => c.skip_field(wt)?,
        }
    }
    Ok(name)
}

/// 从 ValueInfoProto 解出 name（field 1）。其余字段忽略。
fn parse_value_info_name(buf: &[u8]) -> Result<Option<String>> {
    let mut c = Cursor::new(buf);
    let mut name = None;
    while !c.eof() {
        let (field, wt) = c.read_tag()?;
        match (field, wt) {
            (1, 2) => {
                let s = c.read_length_delimited()?;
                name = Some(read_string_field(s)?);
            }
            _ => c.skip_field(wt)?,
        }
    }
    Ok(name)
}

// --- ONNX op_type → OpKind 映射 ---

fn map_op_type(op_type: &str) -> OpKind {
    match op_type {
        "Add" => OpKind::Add,
        "Sub" => OpKind::Sub,
        "Mul" => OpKind::Mul,
        "Div" => OpKind::Div,
        "MatMul" => OpKind::MatMul,
        "Gemm" => OpKind::MatMul, // Gemm ≈ MatMul + bias，简化为 MatMul
        "Relu" => OpKind::Relu,
        "Gelu" => OpKind::Gelu,
        "Sigmoid" => OpKind::Sigmoid,
        "Tanh" => OpKind::Tanh,
        "Softmax" => OpKind::Softmax,
        "LayerNormalization" => OpKind::LayerNorm,
        "InstanceNormalization" => OpKind::LayerNorm,
        "Conv" => OpKind::Conv,
        "MaxPool" | "AveragePool" | "GlobalAveragePool" => OpKind::Pool,
        "Reshape" => OpKind::Reshape,
        "Transpose" => OpKind::Transpose,
        "Concat" => OpKind::Concat,
        "Slice" => OpKind::Slice,
        "Sqrt" => OpKind::Sqrt,
        "Exp" => OpKind::Exp,
        "Pow" => OpKind::Pow,
        "ReduceSum" | "ReduceL1" | "ReduceL2" => OpKind::ReduceSum,
        "ReduceMean" => OpKind::ReduceMean,
        "ReduceMax" => OpKind::ReduceMax,
        // 未知算子 → Custom（attr 记录原始 op_type）
        _ => OpKind::Custom,
    }
}

// --- 名称注册表 ---

struct NameRegistry {
    map: std::collections::HashMap<String, u32>,
}

impl NameRegistry {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
        }
    }
    fn register(&mut self, name: String, v: u32) {
        self.map.entry(name).or_insert(v);
    }
    fn get(&self, name: &str) -> Option<u32> {
        self.map.get(name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手工编码一个最小 ONNX ModelProto：
    /// graph { node { input: "x" output: "y" op_type: "Relu" } input { name: "x" } output { name: "y" } }
    fn build_minimal_onnx() -> Vec<u8> {
        let mut buf = Vec::new();
        // 构造 NodeProto: field1(input)="x", field2(output)="y", field3(op_type)="Relu"
        let node = {
            let mut n = Vec::new();
            write_string_field(&mut n, 1, "x"); // input
            write_string_field(&mut n, 2, "y"); // output
            write_string_field(&mut n, 3, "Relu"); // op_type
            n
        };
        // ValueInfoProto for input "x": field1(name)="x"
        let vi_x = {
            let mut v = Vec::new();
            write_string_field(&mut v, 1, "x");
            v
        };
        // ValueInfoProto for output "y"
        let vi_y = {
            let mut v = Vec::new();
            write_string_field(&mut v, 1, "y");
            v
        };
        // GraphProto: field1(node)=node_buf, field11(input)=vi_x, field12(output)=vi_y
        let graph = {
            let mut g = Vec::new();
            write_len_field(&mut g, 1, &node); // node
            write_len_field(&mut g, 11, &vi_x); // input
            write_len_field(&mut g, 12, &vi_y); // output
            g
        };
        // ModelProto: field7(graph)=graph
        write_len_field(&mut buf, 7, &graph);
        buf
    }

    fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            buf.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        buf.push(v as u8);
    }

    fn write_tag(buf: &mut Vec<u8>, field: u32, wt: u8) {
        write_varint(buf, ((field as u64) << 3) | (wt as u64));
    }

    fn write_string_field(buf: &mut Vec<u8>, field: u32, s: &str) {
        write_tag(buf, field, 2);
        write_varint(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_len_field(buf: &mut Vec<u8>, field: u32, inner: &[u8]) {
        write_tag(buf, field, 2);
        write_varint(buf, inner.len() as u64);
        buf.extend_from_slice(inner);
    }

    #[test]
    fn parses_minimal_onnx() {
        let bytes = build_minimal_onnx();
        let g = parse(&bytes).unwrap();
        assert_eq!(g.name, "onnx");
        // 应有 1 个 Relu 节点
        let kinds: Vec<OpKind> = g.node_ids().map(|id| g.node(id).unwrap().kind).collect();
        assert!(kinds.contains(&OpKind::Relu), "应有 Relu 节点");
        // 应有 1 个图输入 x
        assert_eq!(g.inputs().len(), 1);
        // 应有 1 个图输出 y
        assert_eq!(g.outputs().len(), 1);
    }

    #[test]
    fn empty_input_returns_empty_graph() {
        let g = parse(&[]).unwrap();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn unknown_op_becomes_custom() {
        // 构造一个 op_type="FooBar" 的节点
        let node = {
            let mut n = Vec::new();
            write_string_field(&mut n, 1, "x");
            write_string_field(&mut n, 2, "y");
            write_string_field(&mut n, 3, "FooBar");
            n
        };
        let graph = {
            let mut g = Vec::new();
            write_len_field(&mut g, 1, &node);
            g
        };
        let mut buf = Vec::new();
        write_len_field(&mut buf, 7, &graph);
        let parsed = parse(&buf).unwrap();
        let kinds: Vec<OpKind> = parsed
            .node_ids()
            .map(|id| parsed.node(id).unwrap().kind)
            .collect();
        assert!(kinds.contains(&OpKind::Custom), "未知算子应映射成 Custom");
    }

    #[test]
    fn maps_common_ops() {
        assert_eq!(map_op_type("Add"), OpKind::Add);
        assert_eq!(map_op_type("MatMul"), OpKind::MatMul);
        assert_eq!(map_op_type("Softmax"), OpKind::Softmax);
        assert_eq!(map_op_type("LayerNormalization"), OpKind::LayerNorm);
        assert_eq!(map_op_type("ReduceMean"), OpKind::ReduceMean);
        assert_eq!(map_op_type("WhateverUnknown"), OpKind::Custom);
    }
}
