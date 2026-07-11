//! regalloc — 寄存器分配器
//!
//! 融合方案（最高质量模式）：
//! - 活跃区间分析（Liveness Analysis）：扫描 MachineInstr 序列，计算每个 VReg 的 [def, last_use] 区间
//! - 干扰图构建（Interference Graph）：活跃区间重叠的 VReg 连边（不能共用物理寄存器）
//! - 保守合并（Conservative Briggs Coalescing）：合并 move-related VReg，阈值=3，不引入新溢出
//! - 图着色（Chaitin-Briggs）：simplify（低度节点入栈）→ spill（最小代价节点溢出）→ select（贪心着色）
//! - 分段溢出（Split Spilling）：溢出 VReg 在 def 点插 store，use 点插 load，非全区间
//! - 溢出代价模型：cost = (uses + defs) / interval_size，代价低的优先溢出
//!
//! 设计参考：
//! - Chaitin (1981)：图着色寄存器分配经典算法
//! - Briggs (1994)：保守合并，不引入新溢出
//! - Poletto & Sarkar (1999)：线性扫描（本实现用图着色但活跃区间分析借鉴线性扫描的区间概念）
//! - LLVM greedy allocator：启发式溢出代价 + live range splitting

pub mod allocator;
pub mod coalescing;
pub mod coloring;
pub mod interference;
pub mod liveness;
pub mod types;

pub use allocator::*;
pub use coalescing::*;
pub use coloring::*;
pub use interference::*;
pub use liveness::*;
pub use types::*;

use base::{Graph, NodeId, ValueId};
use isel::Instruction;

/// 将 isel 产出的 Instruction 序列 + IR Graph 的值流信息，
/// 转换为带 VReg operand 的 MachineInstr 序列。
///
/// 每个 IR 值（ValueId）分配一个 VReg。图输入值也分配 VReg。
/// 节点输出 → MachineInstr 的 defs；节点输入 → MachineInstr 的 operands。
///
/// arch::lower 和 isel::select 都保持节点顺序遍历，
/// 所以 instructions[i] 对应 graph 的第 i 个节点。
pub fn lower_to_machine(graph: &Graph, instructions: &[Instruction]) -> Vec<MachineInstr> {
    let mut vreg_map: HashMap<ValueId, VReg> = HashMap::new();
    let mut next_vreg = 0u32;

    // 图输入值分配 VReg
    for &vid in graph.inputs() {
        vreg_map.insert(vid, VReg(next_vreg));
        next_vreg += 1;
    }

    let mut result = Vec::new();

    for (idx, instr) in instructions.iter().enumerate() {
        let node_id = idx as NodeId;
        let node = match graph.node(node_id) {
            Ok(n) => n,
            Err(_) => {
                // 节点不存在（可能是 load/store 等附加 op），用空 operand
                result.push(MachineInstr {
                    op: instr.op.clone(),
                    operands: Vec::new(),
                    defs: Vec::new(),
                    args: instr.args.clone(),
                });
                continue;
            }
        };

        // 输入值 → VReg operand
        let mut operands = Vec::new();
        for &vid in node.inputs() {
            let vreg = vreg_map.entry(vid).or_insert_with(|| {
                let v = VReg(next_vreg);
                next_vreg += 1;
                v
            });
            operands.push(Operand::VReg(*vreg));
        }

        // 输出值 → 新 VReg def
        let mut defs = Vec::new();
        for &vid in node.outputs() {
            let vreg = VReg(next_vreg);
            next_vreg += 1;
            vreg_map.insert(vid, vreg);
            defs.push(Operand::VReg(vreg));
        }

        result.push(MachineInstr {
            op: instr.op.clone(),
            operands,
            defs,
            args: instr.args.clone(),
        });
    }

    result
}

use std::collections::HashMap;
