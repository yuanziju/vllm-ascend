//! IR 图结构 — 基于 raw::Graph 构建的 Region/Block 存储。

use std::sync::atomic::{AtomicU32, Ordering};
use crate::ir::op::Op;
pub use crate::ir::op::ValueId;

static NEXT_MODULE_ID: AtomicU32 = AtomicU32::new(0);
static NEXT_FUNC_ID: AtomicU32 = AtomicU32::new(0);
static NEXT_REGION_ID: AtomicU32 = AtomicU32::new(0);
static NEXT_BLOCK_ID: AtomicU32 = AtomicU32::new(0);
#[allow(dead_code)]
static NEXT_VALUE_ID: AtomicU32 = AtomicU32::new(0);

fn next_id(counter: &AtomicU32) -> u32 { counter.fetch_add(1, Ordering::Relaxed) }

// ---- ID 类型 ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

// ---- Op 引用 ----

pub type OpRef = Box<dyn Op>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    Tensor,
    Scalar,
    Function,
    None,
}

// ---- Terminator ----

#[derive(Debug, Clone)]
pub enum Terminator {
    Return(Option<ValueId>),
    Branch(RegionId, Vec<ValueId>),
    CondBranch(ValueId, RegionId, RegionId, Vec<ValueId>, Vec<ValueId>),
}

// ---- Block ----

#[derive(Debug)]
pub struct BlockData {
    pub id: BlockId,
    pub label: String,
    pub ops: Vec<OpRef>,
    pub arguments: Vec<(ValueId, IrType)>,
    pub terminator: Option<Terminator>,
}

impl Clone for BlockData {
    fn clone(&self) -> Self {
        BlockData {
            id: self.id,
            label: self.label.clone(),
            ops: self.ops.iter().map(|op| op.clone_box()).collect(),
            arguments: self.arguments.clone(),
            terminator: self.terminator.clone(),
        }
    }
}

impl BlockData {
    pub fn new(label: impl Into<String>) -> Self {
        BlockData {
            id: BlockId(next_id(&NEXT_BLOCK_ID)),
            label: label.into(),
            ops: Vec::new(),
            arguments: Vec::new(),
            terminator: None,
        }
    }
}

// ---- Region ----

#[derive(Debug, Clone)]
pub struct RegionData {
    pub id: RegionId,
    pub name: String,
    pub blocks: Vec<BlockData>,
}

impl RegionData {
    pub fn new(name: impl Into<String>) -> Self {
        RegionData {
            id: RegionId(next_id(&NEXT_REGION_ID)),
            name: name.into(),
            blocks: Vec::new(),
        }
    }

    pub fn add_block(&mut self, block: BlockData) -> BlockId {
        let id = block.id;
        self.blocks.push(block);
        id
    }
}

// ---- Function ----

#[derive(Debug, Clone)]
pub struct FunctionData {
    pub id: FunctionId,
    pub name: String,
    pub body: RegionData,
    pub input_types: Vec<IrType>,
    pub output_types: Vec<IrType>,
}

impl FunctionData {
    pub fn new(name: impl Into<String>) -> Self {
        FunctionData {
            id: FunctionId(next_id(&NEXT_FUNC_ID)),
            name: name.into(),
            body: RegionData::new("entry"),
            input_types: Vec::new(),
            output_types: Vec::new(),
        }
    }
}

// ---- Module ----

#[derive(Debug, Clone)]
pub struct ModuleData {
    pub id: ModuleId,
    pub name: String,
    pub functions: Vec<FunctionData>,
}

impl ModuleData {
    pub fn new(name: impl Into<String>) -> Self {
        ModuleData {
            id: ModuleId(next_id(&NEXT_MODULE_ID)),
            name: name.into(),
            functions: Vec::new(),
        }
    }
}

// ---- IrGraph ----

pub struct IrGraph {
    pub modules: Vec<ModuleData>,
}

impl IrGraph {
    pub fn new() -> Self {
        IrGraph { modules: Vec::new() }
    }

    pub fn add_module(&mut self, name: &str) -> ModuleId {
        let module = ModuleData::new(name);
        let id = module.id;
        self.modules.push(module);
        id
    }

    pub fn module(&self, id: ModuleId) -> Option<&ModuleData> {
        self.modules.iter().find(|m| m.id == id)
    }

    pub fn module_mut(&mut self, id: ModuleId) -> Option<&mut ModuleData> {
        self.modules.iter_mut().find(|m| m.id == id)
    }

    pub fn add_function(&mut self, module_id: ModuleId, name: &str) -> Option<FunctionId> {
        let module = self.module_mut(module_id)?;
        let func = FunctionData::new(name);
        let id = func.id;
        module.functions.push(func);
        Some(id)
    }

    pub fn add_block(&mut self, func_id: FunctionId, module_id: ModuleId, label: &str) -> Option<BlockId> {
        let module = self.module_mut(module_id)?;
        let func = module.functions.iter_mut().find(|f| f.id == func_id)?;
        let block = BlockData::new(label);
        let id = block.id;
        func.body.blocks.push(block);
        Some(id)
    }

    pub fn modules_iter(&self) -> std::slice::Iter<'_, ModuleData> {
        self.modules.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_graph_build() {
        let mut graph = IrGraph::new();
        let m = graph.add_module("test_module");
        let f = graph.add_function(m, "test_func").unwrap();
        let b = graph.add_block(f, m, "bb0").unwrap();
        assert_eq!(graph.modules.len(), 1);
        assert_eq!(graph.modules[0].functions.len(), 1);
        assert_eq!(graph.modules[0].functions[0].body.blocks.len(), 1);
        assert_eq!(graph.modules[0].functions[0].body.blocks[0].id, b);
    }

    #[test]
    fn test_region_graph_crud() {
        let mut region = RegionData::new("test");
        let block = BlockData::new("bb0");
        let id = region.add_block(block);
        assert_eq!(region.blocks.len(), 1);
        assert_eq!(region.blocks[0].id, id);
    }

    #[test]
    fn test_block_graph() {
        let mut block = BlockData::new("entry");
        block.terminator = Some(Terminator::Return(None));
        assert!(matches!(block.terminator, Some(Terminator::Return(None))));
    }

    #[test]
    fn test_terminator() {
        let t = Terminator::Return(Some(ValueId(42)));
        match t {
            Terminator::Return(Some(v)) => assert_eq!(v.0, 42),
            _ => panic!("expected Return"),
        }
    }
}