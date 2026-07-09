//! base — 公共 trait 与抽象，以及架构无关 IR 的核心定义。
//!
//! 设计哲学（与用户深度讨论后定）：
//! - **范式**：MLIR 风格（一切皆 op + region），统一框架、可嵌套表达控制流
//! - **层次**：分层渐进 lowering（HLO → LLO）
//! - **副作用**：纯函数式 SSA（无副作用，重排自由）
//! - **值流转**：tagged value ID（值 ID 编码类型 tag，省查表）
//! - **类型**：静态类型 + shape 进入类型系统（依赖类型）
//! - **存储**：连续 packed buffer + unsafe + Safe 包装
//!
//! 上层用 [`Graph`]（Safe API），下层委托 [`raw::RawGraph`]（unsafe 高效）。

pub mod raw;

use raw::{AttrKey, RawGraph};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// 错误类型
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum NeutronError {
    #[error("IR 错误: {0}")]
    Ir(String),
    #[error("前端错误: {0}")]
    Frontend(String),
    #[error("优化错误: {0}")]
    Opt(String),
    #[error("后端/lowering 错误: {0}")]
    Backend(String),
    #[error("指令选择错误: {0}")]
    Isel(String),
    #[error("Lisp 解释器错误: {0}")]
    Lisp(String),
    #[error("其他: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, NeutronError>;

// ---------------------------------------------------------------------------
// ID 类型
// ---------------------------------------------------------------------------

pub type NodeId = u32;
pub type ValueId = u32;

// ---------------------------------------------------------------------------
// 类型系统（静态类型 + shape 进入类型系统）
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I64 = 3,
    I32 = 4,
    Bool = 5,
}

impl DType {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::BF16),
            3 => Ok(Self::I64),
            4 => Ok(Self::I32),
            5 => Ok(Self::Bool),
            _ => Err(NeutronError::Ir(format!("未知 dtype tag: {}", v))),
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeTag {
    ScalarF32 = 0x00,
    ScalarF16 = 0x01,
    ScalarBF16 = 0x02,
    ScalarI64 = 0x03,
    ScalarI32 = 0x04,
    ScalarBool = 0x05,
    TensorF32 = 0x80,
    TensorF16 = 0x81,
    TensorBF16 = 0x82,
    TensorI64 = 0x83,
    TensorI32 = 0x84,
    TensorBool = 0x85,
}

impl TypeTag {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0x00 => Ok(Self::ScalarF32),
            0x01 => Ok(Self::ScalarF16),
            0x02 => Ok(Self::ScalarBF16),
            0x03 => Ok(Self::ScalarI64),
            0x04 => Ok(Self::ScalarI32),
            0x05 => Ok(Self::ScalarBool),
            0x80 => Ok(Self::TensorF32),
            0x81 => Ok(Self::TensorF16),
            0x82 => Ok(Self::TensorBF16),
            0x83 => Ok(Self::TensorI64),
            0x84 => Ok(Self::TensorI32),
            0x85 => Ok(Self::TensorBool),
            _ => Err(NeutronError::Ir(format!("未知 type tag: {:#x}", v))),
        }
    }
    pub fn is_tensor(&self) -> bool {
        (*self as u8) & 0x80 != 0
    }
    pub fn dtype(&self) -> DType {
        DType::from_u8((*self as u8) & 0x7F).unwrap()
    }
    pub fn from_dtype(dt: DType, is_tensor: bool) -> Self {
        let base = dt as u8;
        let v = if is_tensor { base | 0x80 } else { base };
        unsafe { std::mem::transmute(v) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(DType),
    Tensor { dtype: DType, dims: Vec<i64> },
}

impl Type {
    pub fn to_tag(&self) -> TypeTag {
        match self {
            Type::Scalar(dt) => TypeTag::from_dtype(*dt, false),
            Type::Tensor { dtype, .. } => TypeTag::from_dtype(*dtype, true),
        }
    }
}

// ---------------------------------------------------------------------------
// Op Kind
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    MatMul = 4,
    Relu = 5,
    Gelu = 6,
    Sigmoid = 7,
    Tanh = 8,
    Softmax = 9,
    LayerNorm = 10,
    Conv = 11,
    Pool = 12,
    Reshape = 13,
    Transpose = 14,
    Concat = 15,
    Slice = 16,
    Constant = 17,
    Placeholder = 18,
    Return = 19,
    Custom = 64,
}

impl OpKind {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Add),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Mul),
            3 => Ok(Self::Div),
            4 => Ok(Self::MatMul),
            5 => Ok(Self::Relu),
            6 => Ok(Self::Gelu),
            7 => Ok(Self::Sigmoid),
            8 => Ok(Self::Tanh),
            9 => Ok(Self::Softmax),
            10 => Ok(Self::LayerNorm),
            11 => Ok(Self::Conv),
            12 => Ok(Self::Pool),
            13 => Ok(Self::Reshape),
            14 => Ok(Self::Transpose),
            15 => Ok(Self::Concat),
            16 => Ok(Self::Slice),
            17 => Ok(Self::Constant),
            18 => Ok(Self::Placeholder),
            19 => Ok(Self::Return),
            64 => Ok(Self::Custom),
            _ => Err(NeutronError::Ir(format!("未知 op tag: {}", v))),
        }
    }
}

// ---------------------------------------------------------------------------
// 属性（高层 API）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Attr {
    Int(i64),
    Float(f64),
    Bool(bool),
    IntArray(Vec<i64>),
}

// ---------------------------------------------------------------------------
// 节点/值的高层视图
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct NodeView<'a> {
    pub id: NodeId,
    pub kind: OpKind,
    pub raw: &'a RawGraph,
}

impl<'a> NodeView<'a> {
    pub fn inputs(&self) -> &'a [ValueId] {
        self.raw.node_inputs(self.id)
    }
    pub fn outputs(&self) -> &'a [ValueId] {
        self.raw.node_outputs(self.id)
    }
    pub fn attrs(&self) -> &'a [raw::AttrEntry] {
        self.raw.node_attrs(self.id)
    }

    /// 若节点是 Constant 且带 AttrKey::Value (Float) 属性，返回其标量值。
    pub fn constant_value(&self) -> Option<f64> {
        if self.kind != OpKind::Constant {
            return None;
        }
        for e in self.attrs() {
            if e.key == AttrKey::Value as u8 && e.tag == raw::AttrTag::Float as u8 {
                return Some(self.raw.attr_float(e));
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ValueView<'a> {
    pub id: ValueId,
    pub type_tag: TypeTag,
    pub raw: &'a RawGraph,
}

impl<'a> ValueView<'a> {
    pub fn dtype(&self) -> DType {
        self.type_tag.dtype()
    }
    pub fn is_tensor(&self) -> bool {
        self.type_tag.is_tensor()
    }
    pub fn shape(&self) -> &'a [i64] {
        self.raw.value_shape(self.id)
    }
    pub fn name(&self) -> Option<&'a str> {
        self.raw.value_name(self.id)
    }
    pub fn def_node(&self) -> NodeId {
        self.raw.value_def(self.id)
    }
}

// ---------------------------------------------------------------------------
// Graph（Safe API 包装 RawGraph）
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Graph {
    pub name: String,
    pub raw: RawGraph,
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("name", &self.name)
            .field("raw", &self.raw)
            .finish()
    }
}

impl Graph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            raw: RawGraph::new(),
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.raw.node_count()
    }

    #[inline]
    pub fn value_count(&self) -> usize {
        self.raw.value_count()
    }

    #[inline]
    pub fn inputs(&self) -> &[ValueId] {
        &self.raw.inputs
    }

    #[inline]
    pub fn outputs(&self) -> &[ValueId] {
        &self.raw.outputs
    }

    pub fn add_node(&mut self, kind: OpKind) -> NodeId {
        self.raw.alloc_node(kind as u8)
    }

    pub fn add_node_with(
        &mut self,
        kind: OpKind,
        inputs: &[ValueId],
        outputs: &[ValueId],
    ) -> NodeId {
        let id = self.raw.alloc_node(kind as u8);
        self.raw.set_node_inputs(id, inputs);
        self.raw.set_node_outputs(id, outputs);
        id
    }

    pub fn add_value(&mut self, ty: Type, name: Option<&str>, def: NodeId) -> ValueId {
        let tag = ty.to_tag();
        let (rank, shape_off) = match &ty {
            Type::Scalar(_) => (0u8, 0u32),
            Type::Tensor { dims, .. } => {
                let off = self.raw.add_shape(dims);
                (dims.len() as u8, off)
            }
        };
        let name_off = self.raw.add_name(name);
        self.raw
            .alloc_value(tag as u8, rank, shape_off, name_off, def)
    }

    pub fn add_input(&mut self, ty: Type, name: Option<&str>) -> ValueId {
        self.add_value(ty, name, u32::MAX)
    }

    /// 构造一个标量 Constant 节点：节点本身 + 输出 value + AttrKey::Value=float
    pub fn add_constant_f64(&mut self, val: f64) -> (NodeId, ValueId) {
        let node = self.raw.alloc_node(OpKind::Constant as u8);
        let out = self.add_value(Type::Scalar(DType::F32), None, node);
        self.raw.set_node_outputs(node, &[out]);
        self.raw.add_attr_float(node, AttrKey::Value, val);
        (node, out)
    }

    pub fn mark_output(&mut self, v: ValueId) {
        self.raw.outputs.push(v);
    }

    pub fn mark_input(&mut self, v: ValueId) {
        self.raw.inputs.push(v);
    }

    pub fn add_attr(&mut self, node: NodeId, key: AttrKey, attr: Attr) {
        match attr {
            Attr::Int(v) => self.raw.add_attr_int(node, key, v),
            Attr::Float(v) => self.raw.add_attr_float(node, key, v),
            Attr::Bool(v) => self.raw.add_attr_bool(node, key, v),
            Attr::IntArray(v) => self.raw.add_attr_int_array(node, key, &v),
        }
    }

    pub fn node(&self, id: NodeId) -> Result<NodeView<'_>> {
        let hdr = self
            .raw
            .node_hdr
            .get(id as usize)
            .ok_or_else(|| NeutronError::Ir(format!("节点 ID {} 越界", id)))?;
        let kind = OpKind::from_u8(hdr.op_tag)?;
        Ok(NodeView {
            id,
            kind,
            raw: &self.raw,
        })
    }

    pub fn value(&self, id: ValueId) -> Result<ValueView<'_>> {
        let hdr = self
            .raw
            .value_hdr
            .get(id as usize)
            .ok_or_else(|| NeutronError::Ir(format!("值 ID {} 越界", id)))?;
        let type_tag = TypeTag::from_u8(hdr.type_tag)?;
        Ok(ValueView {
            id,
            type_tag,
            raw: &self.raw,
        })
    }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        0..self.node_count() as NodeId
    }

    pub fn value_ids(&self) -> impl Iterator<Item = ValueId> + '_ {
        0..self.value_count() as ValueId
    }

    /// 紧凑化：删除指定节点集合，重建 packed buffer 并 remap 所有 ID。
    ///
    /// 返回新图 + (node_map, value_map)。
    pub fn compact(
        &self,
        remove_nodes: &HashSet<NodeId>,
    ) -> (Graph, HashMap<NodeId, NodeId>, HashMap<ValueId, ValueId>) {
        let mut new_graph = Graph::new(self.name.clone());
        let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::new();

        // 第一遍：复制保留节点的输出 value
        for old_id in self.node_ids() {
            if remove_nodes.contains(&old_id) {
                continue;
            }
            let old_node = self.node(old_id).unwrap();
            let new_node_id = new_graph.add_node(old_node.kind);
            node_map.insert(old_id, new_node_id);

            for &old_vid in old_node.outputs() {
                if let Ok(old_v) = self.value(old_vid) {
                    let ty = if old_v.is_tensor() {
                        Type::Tensor {
                            dtype: old_v.dtype(),
                            dims: old_v.shape().to_vec(),
                        }
                    } else {
                        Type::Scalar(old_v.dtype())
                    };
                    let name = old_v.name().map(|s| s.to_string());
                    let new_vid = new_graph.add_value(ty, name.as_deref(), new_node_id);
                    value_map.insert(old_vid, new_vid);
                }
            }
        }

        // 第二遍：复制图输入 value（无定义节点的）
        for &old_vid in self.inputs() {
            if value_map.contains_key(&old_vid) {
                continue;
            }
            if let Ok(old_v) = self.value(old_vid) {
                let ty = if old_v.is_tensor() {
                    Type::Tensor {
                        dtype: old_v.dtype(),
                        dims: old_v.shape().to_vec(),
                    }
                } else {
                    Type::Scalar(old_v.dtype())
                };
                let name = old_v.name().map(|s| s.to_string());
                let new_vid = new_graph.add_value(ty, name.as_deref(), u32::MAX);
                value_map.insert(old_vid, new_vid);
                new_graph.mark_input(new_vid);
            }
        }

        // 第三遍：重连 inputs/outputs
        for (&old_id, &new_id) in &node_map {
            let old_node = self.node(old_id).unwrap();
            let new_inputs: Vec<ValueId> = old_node
                .inputs()
                .iter()
                .filter_map(|v| value_map.get(v).copied())
                .collect();
            let new_outputs: Vec<ValueId> = old_node
                .outputs()
                .iter()
                .filter_map(|v| value_map.get(v).copied())
                .collect();
            new_graph.raw.set_node_inputs(new_id, &new_inputs);
            new_graph.raw.set_node_outputs(new_id, &new_outputs);
        }

        // 第四遍：图输出重映射
        for &old_vid in self.outputs() {
            if let Some(&new_vid) = value_map.get(&old_vid) {
                new_graph.mark_output(new_vid);
            }
        }

        (new_graph, node_map, value_map)
    }
}

// ---------------------------------------------------------------------------
// Pass / Visitor
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct PassContext {
    pub stats: HashMap<String, usize>,
}

impl PassContext {
    pub fn inc(&mut self, key: impl Into<String>) {
        *self.stats.entry(key.into()).or_insert(0) += 1;
    }
}

pub trait Pass {
    fn name(&self) -> &str;
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()>;
}

pub trait Visitor {
    fn visit_node(&mut self, graph: &Graph, node: NodeView<'_>) -> Result<()>;

    fn visit_graph(&mut self, graph: &Graph) -> Result<()> {
        for id in graph.node_ids() {
            let v = graph.node(id)?;
            self.visit_node(graph, v)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 旧 API 兼容层
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Operation {
    pub kind: OpKind,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Operation {
    pub fn new(kind: OpKind) -> Self {
        Self {
            kind,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    pub id: ValueId,
    pub name: Option<String>,
}

pub use raw::AttrKey as RawAttrKey;
pub use raw::AttrTag as RawAttrTag;
