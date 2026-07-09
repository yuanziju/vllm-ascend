//! arch — 架构描述 + lowering + 目标特定优化

pub mod cuda;
pub mod lowering;
pub mod npu;

use base::{Graph, Result};

/// 架构图（lowering 后）
#[derive(Debug, Default)]
pub struct ArchGraph {
    pub ops: Vec<ArchOp>,
    pub target: common::Target,
}

/// 架构操作（lowering 后的算子）
#[derive(Debug, Clone)]
pub enum ArchOp {
    KernelCall(String),
    Load,
    Store,
}

impl ArchGraph {
    pub fn new(target: common::Target) -> Self {
        Self {
            ops: Vec::new(),
            target,
        }
    }
    pub fn add(&mut self, op: ArchOp) {
        self.ops.push(op);
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// lowering 入口
pub fn lower(graph: &Graph, target: common::Target) -> Result<ArchGraph> {
    lowering::lower(graph, target)
}
