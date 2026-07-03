use std::collections::HashSet;

use crate::graph::ir::{Terminator, ValueId};
use crate::pass::pass::{Pass, PassContext, PassResult};

/// 死代码消除
///
/// 从 Return 和 CondBranch 等终结符开始标记活跃值，反向传播。
/// 移除其输出值没有被任何其他 Op 或终结符使用的 Op。
#[derive(Debug)]
pub struct DCEPass {
    removed: usize,
}

impl DCEPass {
    pub fn new() -> Self {
        DCEPass { removed: 0 }
    }
}

impl Pass for DCEPass {
    fn name(&self) -> &str {
        "dce"
    }

    fn run(&mut self, ctx: &mut PassContext) -> PassResult {
        self.removed = 0;

        let module_count = ctx.graph.modules.len();
        for mi in 0..module_count {
            let func_count = ctx.graph.modules[mi].functions.len();
            for fi in 0..func_count {
                self.eliminate_dead_in_region(mi, fi, ctx);
            }
        }

        if self.removed > 0 {
            PassResult::changed()
        } else {
            PassResult::unchanged()
        }
    }
}

impl DCEPass {
    fn eliminate_dead_in_region(
        &mut self,
        module_idx: usize,
        func_idx: usize,
        ctx: &mut PassContext,
    ) {
        let block_count =
            ctx.graph.modules[module_idx].functions[func_idx].body.blocks.len();
        for bi in 0..block_count {
            self.eliminate_dead_in_block(module_idx, func_idx, bi, ctx);
        }
    }

    fn eliminate_dead_in_block(
        &mut self,
        module_idx: usize,
        func_idx: usize,
        block_idx: usize,
        ctx: &mut PassContext,
    ) {
        let block = &mut ctx.graph.modules[module_idx].functions[func_idx]
            .body.blocks[block_idx];

        let mut live: HashSet<ValueId> = HashSet::new();

        // Mark values used by the terminator
        if let Some(terminator) = &block.terminator {
            Self::mark_terminator_values(terminator, &mut live);
        }

        // Backward propagation: iterate ops in reverse
        for op in block.ops.iter().rev() {
            let outputs = op.outputs();
            let any_output_live = outputs.iter().any(|v| live.contains(v));

            if any_output_live {
                // Mark all input values as live
                for operand in op.inputs() {
                    live.insert(operand.value);
                }
            }
        }

        // Remove ops whose outputs are not live
        // Special case: ops with no outputs (BranchOp, ReturnOp) are always kept
        let before = block.ops.len();
        block.ops.retain(|op| {
            let outputs = op.outputs();
            if outputs.is_empty() {
                true // Keep ops with no outputs (control flow)
            } else {
                outputs.iter().any(|v| live.contains(v))
            }
        });
        self.removed += before - block.ops.len();
    }

    fn mark_terminator_values(terminator: &Terminator, live: &mut HashSet<ValueId>) {
        match terminator {
            Terminator::Return(Some(v)) => {
                live.insert(*v);
            }
            Terminator::Return(None) => {}
            Terminator::Branch(_, args) => {
                live.extend(args);
            }
            Terminator::CondBranch(cond, _, _, args1, args2) => {
                live.insert(*cond);
                live.extend(args1);
                live.extend(args2);
            }
        }
    }
}