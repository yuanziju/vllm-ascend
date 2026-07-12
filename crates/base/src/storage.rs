//! storage — 底层图存储：连续 packed buffer + unsafe。
//!
//! 设计哲学：上层 [`crate::Graph`] 提供 Safe API，下层用巨量 unsafe 构建
//! 丑陋但高效的王国。所有节点/值/属性压入连续 buffer，ID 即偏移量，O(1) 访问。

use std::collections::HashMap;

/// 节点定长头（32 字节，对齐 8）
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NodeHeader {
    pub op_tag: u8,
    pub _pad: [u8; 3],
    pub inputs_off: u32,
    pub inputs_len: u32,
    pub outputs_off: u32,
    pub outputs_len: u32,
    pub attrs_off: u32,
    pub attrs_len: u32,
    pub parent_region: u32,
}

impl std::fmt::Debug for NodeHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeHeader")
            .field("op_tag", &self.op_tag)
            .field("inputs_off", &self.inputs_off)
            .field("inputs_len", &self.inputs_len)
            .field("outputs_off", &self.outputs_off)
            .field("outputs_len", &self.outputs_len)
            .field("attrs_off", &self.attrs_off)
            .field("attrs_len", &self.attrs_len)
            .field("parent_region", &self.parent_region)
            .finish()
    }
}

/// 值定长头（24 字节）
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ValueHeader {
    pub type_tag: u8,
    pub rank: u8,
    pub _pad: [u8; 2],
    pub shape_off: u32,
    pub name_off: u32,
    pub def_node: u32,
}

impl std::fmt::Debug for ValueHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueHeader")
            .field("type_tag", &self.type_tag)
            .field("rank", &self.rank)
            .field("shape_off", &self.shape_off)
            .field("name_off", &self.name_off)
            .field("def_node", &self.def_node)
            .finish()
    }
}

/// 属性键枚举
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttrKey {
    Strides = 0,
    Padding = 1,
    Dilation = 2,
    Groups = 3,
    Axis = 4,
    Alpha = 5,
    Beta = 6,
    TransposeA = 7,
    TransposeB = 8,
    Epsilon = 9,
    Shape = 10,
    /// Constant 节点的标量值（Float tag），用于代数折叠/识别 0/1
    Value = 11,
    /// Transpose 的轴排列序列（IntArray tag），如 `[1, 0, 2]`
    Perm = 12,
    Custom = 255,
}

impl AttrKey {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Strides),
            1 => Some(Self::Padding),
            2 => Some(Self::Dilation),
            3 => Some(Self::Groups),
            4 => Some(Self::Axis),
            5 => Some(Self::Alpha),
            6 => Some(Self::Beta),
            7 => Some(Self::TransposeA),
            8 => Some(Self::TransposeB),
            9 => Some(Self::Epsilon),
            10 => Some(Self::Shape),
            11 => Some(Self::Value),
            12 => Some(Self::Perm),
            255 => Some(Self::Custom),
            _ => None,
        }
    }
}

/// 属性值 tag
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum AttrTag {
    Int = 0,
    Float = 1,
    Bool = 2,
    IntArray = 3,
    FloatArray = 4,
}

impl AttrTag {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Int),
            1 => Some(Self::Float),
            2 => Some(Self::Bool),
            3 => Some(Self::IntArray),
            4 => Some(Self::FloatArray),
            _ => None,
        }
    }
}

/// 属性条目（12 字节定长头 + data 在 attr_data 池）
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AttrEntry {
    pub key: u8,
    pub tag: u8,
    pub _pad: [u8; 2],
    pub data_off: u32,
    pub data_len: u32,
}

impl std::fmt::Debug for AttrEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttrEntry")
            .field("key", &self.key)
            .field("tag", &self.tag)
            .field("data_off", &self.data_off)
            .field("data_len", &self.data_len)
            .finish()
    }
}

/// 底层 packed 图存储
#[derive(Default)]
pub struct StorageGraph {
    pub node_hdr: Vec<NodeHeader>,
    pub value_hdr: Vec<ValueHeader>,
    pub edges: Vec<u32>,
    pub attrs: Vec<AttrEntry>,
    pub attr_data: Vec<u8>,
    pub shape_data: Vec<i64>,
    pub name_data: Vec<u8>,
    pub custom_keys: HashMap<u32, String>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}

impl std::fmt::Debug for StorageGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageGraph")
            .field("node_count", &self.node_hdr.len())
            .field("value_count", &self.value_hdr.len())
            .field("edge_count", &self.edges.len())
            .field("attr_count", &self.attrs.len())
            .field("input_count", &self.inputs.len())
            .field("output_count", &self.outputs.len())
            .finish()
    }
}

impl StorageGraph {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_hdr.len()
    }

    #[inline]
    pub fn value_count(&self) -> usize {
        self.value_hdr.len()
    }

    #[inline]
    pub fn alloc_node(&mut self, op_tag: u8) -> u32 {
        let id = self.node_hdr.len() as u32;
        self.node_hdr.push(NodeHeader {
            op_tag,
            _pad: [0; 3],
            inputs_off: 0,
            inputs_len: 0,
            outputs_off: 0,
            outputs_len: 0,
            attrs_off: 0,
            attrs_len: 0,
            parent_region: u32::MAX,
        });
        id
    }

    #[inline]
    pub fn alloc_value(
        &mut self,
        type_tag: u8,
        rank: u8,
        shape_off: u32,
        name_off: u32,
        def_node: u32,
    ) -> u32 {
        let id = self.value_hdr.len() as u32;
        self.value_hdr.push(ValueHeader {
            type_tag,
            rank,
            _pad: [0; 2],
            shape_off,
            name_off,
            def_node,
        });
        id
    }

    pub fn set_node_inputs(&mut self, node: u32, inputs: &[u32]) {
        let off = self.edges.len() as u32;
        self.edges.extend_from_slice(inputs);
        let hdr = &mut self.node_hdr[node as usize];
        hdr.inputs_off = off;
        hdr.inputs_len = inputs.len() as u32;
    }

    pub fn set_node_outputs(&mut self, node: u32, outputs: &[u32]) {
        let off = self.edges.len() as u32;
        self.edges.extend_from_slice(outputs);
        let hdr = &mut self.node_hdr[node as usize];
        hdr.outputs_off = off;
        hdr.outputs_len = outputs.len() as u32;
    }

    pub fn add_attr_int(&mut self, node: u32, key: AttrKey, val: i64) {
        let data_off = self.attr_data.len() as u32;
        self.attr_data.extend_from_slice(&val.to_le_bytes());
        self.push_attr(
            node,
            AttrEntry {
                key: key as u8,
                tag: AttrTag::Int as u8,
                _pad: [0; 2],
                data_off,
                data_len: 8,
            },
        );
    }

    pub fn add_attr_float(&mut self, node: u32, key: AttrKey, val: f64) {
        let data_off = self.attr_data.len() as u32;
        self.attr_data.extend_from_slice(&val.to_le_bytes());
        self.push_attr(
            node,
            AttrEntry {
                key: key as u8,
                tag: AttrTag::Float as u8,
                _pad: [0; 2],
                data_off,
                data_len: 8,
            },
        );
    }

    pub fn add_attr_bool(&mut self, node: u32, key: AttrKey, val: bool) {
        let data_off = self.attr_data.len() as u32;
        self.attr_data.push(val as u8);
        self.push_attr(
            node,
            AttrEntry {
                key: key as u8,
                tag: AttrTag::Bool as u8,
                _pad: [0; 2],
                data_off,
                data_len: 1,
            },
        );
    }

    pub fn add_attr_int_array(&mut self, node: u32, key: AttrKey, vals: &[i64]) {
        let data_off = self.attr_data.len() as u32;
        for v in vals {
            self.attr_data.extend_from_slice(&v.to_le_bytes());
        }
        self.push_attr(
            node,
            AttrEntry {
                key: key as u8,
                tag: AttrTag::IntArray as u8,
                _pad: [0; 2],
                data_off,
                data_len: (vals.len() * 8) as u32,
            },
        );
    }

    pub fn add_attr_float_array(&mut self, node: u32, key: AttrKey, vals: &[f64]) {
        let data_off = self.attr_data.len() as u32;
        for v in vals {
            self.attr_data.extend_from_slice(&v.to_le_bytes());
        }
        self.push_attr(
            node,
            AttrEntry {
                key: key as u8,
                tag: AttrTag::FloatArray as u8,
                _pad: [0; 2],
                data_off,
                data_len: (vals.len() * 8) as u32,
            },
        );
    }

    #[inline]
    fn push_attr(&mut self, node: u32, entry: AttrEntry) {
        let off = self.attrs.len() as u32;
        self.attrs.push(entry);
        let hdr = &mut self.node_hdr[node as usize];
        if hdr.attrs_len == 0 {
            hdr.attrs_off = off;
        }
        hdr.attrs_len += 1;
    }

    pub fn add_shape(&mut self, dims: &[i64]) -> u32 {
        let off = self.shape_data.len() as u32;
        self.shape_data.extend_from_slice(dims);
        off
    }

    pub fn add_name(&mut self, name: Option<&str>) -> u32 {
        match name {
            None => u32::MAX,
            Some(s) => {
                let off = self.name_data.len() as u32;
                self.name_data.extend_from_slice(s.as_bytes());
                self.name_data.push(0);
                off
            }
        }
    }

    #[inline]
    pub fn node_inputs(&self, node: u32) -> &[u32] {
        let h = &self.node_hdr[node as usize];
        &self.edges[h.inputs_off as usize..(h.inputs_off + h.inputs_len) as usize]
    }

    #[inline]
    pub fn node_outputs(&self, node: u32) -> &[u32] {
        let h = &self.node_hdr[node as usize];
        &self.edges[h.outputs_off as usize..(h.outputs_off + h.outputs_len) as usize]
    }

    #[inline]
    pub fn node_attrs(&self, node: u32) -> &[AttrEntry] {
        let h = &self.node_hdr[node as usize];
        &self.attrs[h.attrs_off as usize..(h.attrs_off + h.attrs_len) as usize]
    }

    #[inline]
    pub fn attr_int(&self, entry: &AttrEntry) -> i64 {
        debug_assert_eq!(entry.tag, AttrTag::Int as u8);
        let bytes: [u8; 8] = self.attr_data[entry.data_off as usize..(entry.data_off + 8) as usize]
            .try_into()
            .unwrap();
        i64::from_le_bytes(bytes)
    }

    #[inline]
    pub fn attr_float(&self, entry: &AttrEntry) -> f64 {
        debug_assert_eq!(entry.tag, AttrTag::Float as u8);
        let bytes: [u8; 8] = self.attr_data[entry.data_off as usize..(entry.data_off + 8) as usize]
            .try_into()
            .unwrap();
        f64::from_le_bytes(bytes)
    }

    #[inline]
    pub fn attr_bool(&self, entry: &AttrEntry) -> bool {
        debug_assert_eq!(entry.tag, AttrTag::Bool as u8);
        self.attr_data[entry.data_off as usize] != 0
    }

    /// 读取属性 int array。
    ///
    /// # Safety
    /// 前提：attr_data 是 `Vec<u8>`，对齐 1。强转 `&[i64]` 需对齐 8。
    /// 生产环境应保证 attr_data 按 8 对齐分配。此处简化用 unsafe。
    #[inline]
    pub fn attr_int_array(&self, entry: &AttrEntry) -> &[i64] {
        debug_assert_eq!(entry.tag, AttrTag::IntArray as u8);
        let start = entry.data_off as usize;
        let end = start + entry.data_len as usize;
        let bytes = &self.attr_data[start..end];
        let count = bytes.len() / 8;
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i64, count) }
    }

    /// 读取属性 float array（每个元素 8 字节 LE f64）。
    ///
    /// # Safety
    /// 同 `attr_int_array`，强转 &[f64] 需对齐 8。
    #[inline]
    pub fn attr_float_array(&self, entry: &AttrEntry) -> &[f64] {
        debug_assert_eq!(entry.tag, AttrTag::FloatArray as u8);
        let start = entry.data_off as usize;
        let end = start + entry.data_len as usize;
        let bytes = &self.attr_data[start..end];
        let count = bytes.len() / 8;
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, count) }
    }

    #[inline]
    pub fn value_shape(&self, value: u32) -> &[i64] {
        let h = &self.value_hdr[value as usize];
        &self.shape_data[h.shape_off as usize..(h.shape_off + h.rank as u32) as usize]
    }

    /// 设置 value 的 shape（更新 rank + shape_off，追加到 shape_data 池）。
    /// 用于 shape 推断 pass 回填未知 shape（rank/shape_off 可变）。
    pub fn set_value_shape(&mut self, value: u32, dims: &[i64]) {
        let new_off = self.add_shape(dims);
        let h = &mut self.value_hdr[value as usize];
        h.rank = dims.len() as u8;
        h.shape_off = new_off;
    }

    pub fn value_name(&self, value: u32) -> Option<&str> {
        let h = &self.value_hdr[value as usize];
        if h.name_off == u32::MAX {
            return None;
        }
        let start = h.name_off as usize;
        let end = self.name_data[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| start + p)
            .unwrap_or(self.name_data.len());
        std::str::from_utf8(&self.name_data[start..end]).ok()
    }

    #[inline]
    pub fn value_def(&self, value: u32) -> u32 {
        self.value_hdr[value as usize].def_node
    }
}
