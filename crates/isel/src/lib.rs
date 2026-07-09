//! isel — 指令选择（基于 lisp 规则）

use arch::{ArchGraph, ArchOp};
use base::Result;

/// 最终指令
#[derive(Debug, Clone)]
pub struct Instruction {
    pub op: String,
    pub args: Vec<String>,
}

/// 从 ArchGraph 选择指令
pub fn select(arch_graph: &ArchGraph) -> Result<Vec<Instruction>> {
    let mut instrs = Vec::new();
    for op in &arch_graph.ops {
        match op {
            ArchOp::KernelCall(name) => {
                instrs.push(Instruction {
                    op: name.clone(),
                    args: Vec::new(),
                });
            }
            ArchOp::Load => {
                instrs.push(Instruction {
                    op: "load".into(),
                    args: Vec::new(),
                });
            }
            ArchOp::Store => {
                instrs.push(Instruction {
                    op: "store".into(),
                    args: Vec::new(),
                });
            }
        }
    }
    Ok(instrs)
}
