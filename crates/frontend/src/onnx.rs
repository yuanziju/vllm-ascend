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
//! 属性解析：NodeProto.attribute (field 5, repeated AttributeProto) 解出
//! name + value（int/float/ints），按 op_type 喂给对应的 StorageAttrKey：
//! - reduce/concat 的 axis/axes → AttrKey::Axis
//! - LayerNormalization 的 epsilon → AttrKey::Epsilon
//! - Transpose 的 perm → AttrKey::Perm
//! - Reshape 的 shape（attr 形式）→ AttrKey::Shape
//!
//! 其余属性暂忽略（前向兼容，不报错）。
//!
//! Protobuf 字段编号参考 ONNX schema：
//! - ModelProto: 7=graph(LEN)
//! - GraphProto: 1=node(repeated NodeProto), 2=name(string), 5=initializer(repeated)
//! - NodeProto: 1=input(repeated string), 2=output(repeated string), 3=op_type(string),
//!   4=name(string), 5=attribute(repeated AttributeProto), 7=domain(string)
//! - AttributeProto: 1=name(string), 3=type(varint), 4=f(FIXED32 float),
//!   5=i(varint int64), 6=s(bytes), 7=t(TensorProto), 20=floats(packed),
//!   21=ints(packed repeated int64)

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

        // 按 op_type 把 ONNX 属性喂给对应的 StorageAttrKey
        apply_attributes(&mut g, nid, kind, &node.attributes);
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

/// 按 op_type 把解析出的 ONNX 属性写到对应节点的 StorageAttrKey。
/// 不识别的属性静默忽略（前向兼容）。
fn apply_attributes(g: &mut Graph, nid: base::NodeId, kind: OpKind, attrs: &[AttrInfo]) {
    for attr in attrs {
        match (kind, attr.name.as_str(), &attr.value) {
            // reduce/concat 的 axis（INT）→ AttrKey::Axis
            (
                OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax | OpKind::Concat,
                "axis",
                AttrValue::Int(v),
            ) => {
                g.storage.add_attr_int(nid, StorageAttrKey::Axis, *v);
            }
            // ONNX ReduceSum 用 "axes"（INTS），取首元素作单一轴
            (
                OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax,
                "axes",
                AttrValue::Ints(vs),
            ) if !vs.is_empty() => {
                g.storage.add_attr_int(nid, StorageAttrKey::Axis, vs[0]);
            }
            // LayerNormalization 的 epsilon（FLOAT）→ AttrKey::Epsilon
            (OpKind::LayerNorm, "epsilon", AttrValue::Float(v)) => {
                g.storage.add_attr_float(nid, StorageAttrKey::Epsilon, *v);
            }
            // Transpose 的 perm（INTS）→ AttrKey::Perm
            (OpKind::Transpose, "perm", AttrValue::Ints(vs)) => {
                g.storage.add_attr_int_array(nid, StorageAttrKey::Perm, vs);
            }
            // Reshape 的 shape（attr 形式，INTS）→ AttrKey::Shape
            (OpKind::Reshape, "shape", AttrValue::Ints(vs)) => {
                g.storage.add_attr_int_array(nid, StorageAttrKey::Shape, vs);
            }
            _ => {}
        }
    }
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
    attributes: Vec<AttrInfo>,
}

/// AttributeProto 解析结果。type 字段不存（按 value 字段存在性推断）。
#[derive(Debug, Default)]
struct AttrInfo {
    name: String,
    value: AttrValue,
}

#[derive(Debug, Default)]
enum AttrValue {
    #[default]
    None,
    Int(i64),
    Float(f64),
    Ints(Vec<i64>),
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
            // NodeProto.attribute = field 5, repeated AttributeProto (LEN)
            (5, 2) => {
                let attr_buf = c.read_length_delimited()?;
                n.attributes.push(parse_attribute(attr_buf)?);
            }
            // NodeProto.name = field 4, string（跳过，用 op_type）
            // NodeProto.domain = field 7（跳过）
            _ => c.skip_field(wt)?,
        }
    }
    Ok(n)
}

/// 解析 AttributeProto：name(1) + type(3,跳过) + f(4,FIXED32) + i(5,varint)
/// + s(6,跳过) + t(7,跳过) + floats(20,跳过) + ints(21,packed/non-packed)。
///
/// value 按 i/f/ints 存在性推断（i 优先，其次 f，其次 ints）。
fn parse_attribute(buf: &[u8]) -> Result<AttrInfo> {
    let mut c = Cursor::new(buf);
    let mut name = String::new();
    let mut int_val: Option<i64> = None;
    let mut float_val: Option<f64> = None;
    let mut ints_val: Vec<i64> = Vec::new();
    while !c.eof() {
        let (field, wt) = c.read_tag()?;
        match (field, wt) {
            (1, 2) => {
                let s = c.read_length_delimited()?;
                name = read_string_field(s)?;
            }
            // type (varint) - 跳过，按 value 字段存在性推断
            (3, 0) => {
                c.read_varint()?;
            }
            // f (float, FIXED32) - 4 字节 LE f32
            (4, 5) => {
                if c.pos + 4 > c.data.len() {
                    return Err(base::NeutronError::Frontend("AttributeProto.f 越界".into()));
                }
                let bytes = [
                    c.data[c.pos],
                    c.data[c.pos + 1],
                    c.data[c.pos + 2],
                    c.data[c.pos + 3],
                ];
                c.pos += 4;
                float_val = Some(f32::from_le_bytes(bytes) as f64);
            }
            // i (int64, varint)
            (5, 0) => {
                int_val = Some(c.read_varint()? as i64);
            }
            // s (bytes), t (TensorProto), g (GraphProto) - 跳过
            (6, 2) | (7, 2) | (8, 2) => {
                c.read_length_delimited()?;
            }
            // floats (repeated float, packed LEN) - 跳过（暂不用）
            (20, 2) => {
                c.read_length_delimited()?;
            }
            // floats 非打包单元素（legacy, FIXED32）
            (20, 5) => {
                c.pos += 4;
            }
            // ints (repeated int64, packed LEN)
            (21, 2) => {
                let buf2 = c.read_length_delimited()?;
                let mut c2 = Cursor::new(buf2);
                while !c2.eof() {
                    ints_val.push(c2.read_varint()? as i64);
                }
            }
            // ints 非打包单元素（legacy, varint）
            (21, 0) => {
                ints_val.push(c.read_varint()? as i64);
            }
            _ => c.skip_field(wt)?,
        }
    }
    let value = if let Some(v) = int_val {
        AttrValue::Int(v)
    } else if let Some(v) = float_val {
        AttrValue::Float(v)
    } else if !ints_val.is_empty() {
        AttrValue::Ints(ints_val)
    } else {
        AttrValue::None
    };
    Ok(AttrInfo { name, value })
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

    // --- 属性解析测试辅助 ---

    /// 写一个 FIXED32 字段（protobuf float，wire_type=5）
    fn write_fixed32_field(buf: &mut Vec<u8>, field: u32, val: f32) {
        write_tag(buf, field, 5);
        buf.extend_from_slice(&val.to_le_bytes());
    }

    /// 构造 AttributeProto（INT 类型）：name + i(value)
    fn build_attr_int(name: &str, value: i64) -> Vec<u8> {
        let mut a = Vec::new();
        write_string_field(&mut a, 1, name); // name
        write_tag(&mut a, 5, 0); // field=5(i), wt=0(varint)
        write_varint(&mut a, value as u64);
        a
    }

    /// 构造 AttributeProto（FLOAT 类型）：name + f(value as f32, FIXED32)
    fn build_attr_float(name: &str, value: f32) -> Vec<u8> {
        let mut a = Vec::new();
        write_string_field(&mut a, 1, name); // name
        write_fixed32_field(&mut a, 4, value); // f (field 4, FIXED32)
        a
    }

    /// 构造 AttributeProto（INTS 类型）：name + ints(packed)
    fn build_attr_ints(name: &str, values: &[i64]) -> Vec<u8> {
        let mut a = Vec::new();
        write_string_field(&mut a, 1, name); // name
                                             // packed ints: field 21, LEN，payload = 各 varint 拼接
        let mut payload = Vec::new();
        for &v in values {
            write_varint(&mut payload, v as u64);
        }
        write_len_field(&mut a, 21, &payload);
        a
    }

    /// 构造含若干属性的 NodeProto
    fn build_node_with_attrs(
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut n = Vec::new();
        for &i in inputs {
            write_string_field(&mut n, 1, i);
        }
        for &o in outputs {
            write_string_field(&mut n, 2, o);
        }
        write_string_field(&mut n, 3, op_type);
        for attr in attrs {
            write_len_field(&mut n, 5, attr); // NodeProto.attribute = field 5
        }
        n
    }

    fn read_axis_attr(g: &Graph, nid: base::NodeId) -> Option<i64> {
        for e in g.node(nid).ok()?.attrs() {
            if e.key == base::StorageAttrKey::Axis as u8
                && e.tag == base::storage::AttrTag::Int as u8
            {
                return Some(g.node(nid).unwrap().storage.attr_int(e));
            }
        }
        None
    }

    fn read_epsilon_attr(g: &Graph, nid: base::NodeId) -> Option<f64> {
        for e in g.node(nid).ok()?.attrs() {
            if e.key == base::StorageAttrKey::Epsilon as u8
                && e.tag == base::storage::AttrTag::Float as u8
            {
                return Some(g.node(nid).unwrap().storage.attr_float(e));
            }
        }
        None
    }

    fn read_perm_attr(g: &Graph, nid: base::NodeId) -> Option<Vec<i64>> {
        for e in g.node(nid).ok()?.attrs() {
            if e.key == base::StorageAttrKey::Perm as u8
                && e.tag == base::storage::AttrTag::IntArray as u8
            {
                return Some(g.node(nid).unwrap().storage.attr_int_array(e).to_vec());
            }
        }
        None
    }

    #[test]
    fn parses_reduce_axes_attribute() {
        // ReduceMean(x, axes=[1]) → ReduceMean 节点带 Axis=1
        let attr = build_attr_ints("axes", &[1]);
        let node = build_node_with_attrs("ReduceMean", &["x"], &["y"], &[attr]);
        let graph = {
            let mut g = Vec::new();
            write_len_field(&mut g, 1, &node);
            g
        };
        let mut buf = Vec::new();
        write_len_field(&mut buf, 7, &graph);
        let g = parse(&buf).unwrap();
        // 找到 ReduceMean 节点
        let rs: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::ReduceMean)
            .collect();
        assert_eq!(rs.len(), 1);
        assert_eq!(
            read_axis_attr(&g, rs[0]),
            Some(1),
            "axes=[1] 应映射到 Axis=1"
        );
    }

    #[test]
    fn parses_concat_axis_attribute() {
        // Concat(inputs, axis=0) → Concat 节点带 Axis=0
        let attr = build_attr_int("axis", 0);
        let node = build_node_with_attrs("Concat", &["a", "b"], &["y"], &[attr]);
        let graph = {
            let mut g = Vec::new();
            write_len_field(&mut g, 1, &node);
            g
        };
        let mut buf = Vec::new();
        write_len_field(&mut buf, 7, &graph);
        let g = parse(&buf).unwrap();
        let cc: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::Concat)
            .collect();
        assert_eq!(cc.len(), 1);
        assert_eq!(read_axis_attr(&g, cc[0]), Some(0), "axis=0 应映射到 Axis=0");
    }

    #[test]
    fn parses_layernorm_epsilon_attribute() {
        // LayerNormalization(x, epsilon=1e-5) → LayerNorm 节点带 Epsilon=1e-5
        let attr = build_attr_float("epsilon", 1e-5);
        let node = build_node_with_attrs("LayerNormalization", &["x"], &["y"], &[attr]);
        let graph = {
            let mut g = Vec::new();
            write_len_field(&mut g, 1, &node);
            g
        };
        let mut buf = Vec::new();
        write_len_field(&mut buf, 7, &graph);
        let g = parse(&buf).unwrap();
        let ln: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::LayerNorm)
            .collect();
        assert_eq!(ln.len(), 1);
        let eps = read_epsilon_attr(&g, ln[0]).expect("应有 Epsilon attr");
        assert!((eps - 1e-5).abs() < 1e-9, "epsilon 应为 1e-5，实际 {eps}");
    }

    #[test]
    fn parses_transpose_perm_attribute() {
        // Transpose(x, perm=[1,0,2]) → Transpose 节点带 Perm=[1,0,2]
        let attr = build_attr_ints("perm", &[1, 0, 2]);
        let node = build_node_with_attrs("Transpose", &["x"], &["y"], &[attr]);
        let graph = {
            let mut g = Vec::new();
            write_len_field(&mut g, 1, &node);
            g
        };
        let mut buf = Vec::new();
        write_len_field(&mut buf, 7, &graph);
        let g = parse(&buf).unwrap();
        let tp: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::Transpose)
            .collect();
        assert_eq!(tp.len(), 1);
        assert_eq!(
            read_perm_attr(&g, tp[0]),
            Some(vec![1, 0, 2]),
            "perm=[1,0,2] 应映射到 Perm=[1,0,2]"
        );
    }

    #[test]
    fn unknown_attribute_ignored() {
        // ReduceMean 带未知属性 "keepdims"（INT）应静默忽略，不报错
        let attr = build_attr_int("keepdims", 1);
        let node = build_node_with_attrs("ReduceMean", &["x"], &["y"], &[attr]);
        let graph = {
            let mut g = Vec::new();
            write_len_field(&mut g, 1, &node);
            g
        };
        let mut buf = Vec::new();
        write_len_field(&mut buf, 7, &graph);
        // 不应报错
        let g = parse(&buf).unwrap();
        let rs: Vec<_> = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::ReduceMean)
            .collect();
        assert_eq!(rs.len(), 1);
        // keepdims 不映射到任何 attr，故 Axis 应为 None
        assert_eq!(read_axis_attr(&g, rs[0]), None, "未知属性应被忽略");
    }
}
