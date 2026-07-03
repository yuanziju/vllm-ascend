use std::collections::HashMap;

use crate::graph::ir::ValueId;
use crate::pass::pass::{Pass, PassContext, PassResult};

/// 公共子表达式消除
///
/// 在同一个 Block 内，如果两个 Op 的 dialect+name 相同且输入相同，消除后一个。
/// 消除时将后者的所有引用替换为前者的输出，然后删除后者。
#[derive(Debug)]
pub struct CSEPass {
    eliminated: usize,
}

impl CSEPass {
    pub fn new() -> Self {
        CSEPass { eliminated: 0 }
    }
}

impl Pass for CSEPass {
    fn name(&self) -> &str {
        "cse"
    }

    fn run(&mut self, ctx: &mut PassContext) -> PassResult {
        self.eliminated = 0;

        let module_count = ctx.graph.modules.len();
        for mi in 0..module_count {
            let func_count = ctx.graph.modules[mi].functions.len();
            for fi in 0..func_count {
                self.eliminate_cse_in_region(mi, fi, ctx);
            }
        }

        if self.eliminated > 0 {
            PassResult::changed()
        } else {
            PassResult::unchanged()
        }
    }
}

impl CSEPass {
    fn eliminate_cse_in_region(
        &mut self,
        module_idx: usize,
        func_idx: usize,
        ctx: &mut PassContext,
    ) {
        let block_count =
            ctx.graph.modules[module_idx].functions[func_idx].body.blocks.len();
        for bi in 0..block_count {
            self.eliminate_cse_in_block(module_idx, func_idx, bi, ctx);
        }
    }

    fn eliminate_cse_in_block(
        &mut self,
        module_idx: usize,
        func_idx: usize,
        block_idx: usize,
        ctx: &mut PassContext,
    ) {
        let block = &mut ctx.graph.modules[module_idx].functions[func_idx]
            .body.blocks[block_idx];

        // Build: op_key -> (first_index, outputs)
        let mut seen: HashMap<(String, Vec<ValueId>), (usize, Vec<ValueId>)> = HashMap::new();
        let mut dup_indices: Vec<usize> = Vec::new();
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::new();

        for (i, op) in block.ops.iter().enumerate() {
            let key = op.op_key();
            if let Some(&(_, ref orig_outputs)) = seen.get(&key) {
                // Duplicate found: map its outputs to the original's outputs
                let dup_outputs = op.outputs().to_vec();
                for (dup, orig) in dup_outputs.iter().zip(orig_outputs.iter()) {
                    value_map.insert(*dup, *orig);
                }
                dup_indices.push(i);
            } else {
                seen.insert(key, (i, op.outputs().to_vec()));
            }
        }

        if dup_indices.is_empty() {
            return;
        }

        // Replace all operand references in remaining ops
        for op in &mut block.ops {
            for operand in op.inputs_mut() {
                if let Some(new_val) = value_map.get(&operand.value) {
                    operand.value = *new_val;
                }
            }
        }

        // Remove duplicate ops (iterate in reverse to preserve indices)
        let mut new_ops: Vec<_> = Vec::with_capacity(block.ops.len() - dup_indices.len());
        let dup_set: std::collections::HashSet<usize> = dup_indices.iter().copied().collect();
        for (i, op) in block.ops.drain(..).enumerate() {
            if !dup_set.contains(&i) {
                new_ops.push(op);
            }
        }
        let eliminated = dup_indices.len();
        block.ops = new_ops;
        self.eliminated += eliminated;
    }
}