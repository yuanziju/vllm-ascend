//! 寄存器分配（regalloc）。
//!
//! 输入：isel 产出的 Vec<Instruction>（带 VReg operand）+ ArchGraph（含 DeviceDesc 寄存器上限）。
//! 输出：VReg → PReg 映射表（Allocation），不改写 IR（方式 X：保住 SSA 重排自由）。
//!
//! 四档模式（RegAllocMode）：
//! - Fast：线性扫描（Poletto-Sarkar），O(n log n)，质量够用
//! - Standard：SSA 消φ + IRC（图着色 + Coalescing 迭代），兼顾时间下的最优【后续实现】
//! - Quality：Standard + ML IR 领域特化增强【后续实现】
//! - Exhaustive：暴力枚举，彩蛋/教学用【后续实现】
//!
//! 本轮实现：crate 骨架 + live range 分析 + Fast 模式（线性扫描）。

use std::collections::HashMap;

use arch::{ArchGraph, DeviceDesc};
use common::RegAllocMode;
use isel::{Instruction, Operand, VReg};

/// 物理寄存器 ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PReg(pub u32);

/// 寄存器分配结果：VReg → PReg 映射
#[derive(Debug, Clone)]
pub struct Allocation {
    /// VReg 到 PReg 的映射
    pub vreg_to_preg: HashMap<VReg, PReg>,
    /// 被溢出到内存的 VReg 列表
    pub spilled: Vec<VReg>,
}

/// 寄存器分配错误
#[derive(Debug, thiserror::Error)]
pub enum RegAllocError {
    #[error("虚拟寄存器数量超出物理寄存器且无法溢出: {0}")]
    OutOfRegisters(String),
}

pub type Result<T> = std::result::Result<T, RegAllocError>;

/// 虚拟寄存器的活跃区间（linear scan 用）。
/// start = 第一次 def 的指令索引，end = 最后一次 use 的指令索引。
#[derive(Debug, Clone, Copy)]
struct LiveInterval {
    vreg: VReg,
    start: usize,
    end: usize,
}

/// 计算 Vec<Instruction> 的所有 VReg 的活跃区间。
///
/// SSA 下每个 VReg 理论上只 def 一次，此处仍保守取 start=min(def 点)、
/// end=max(use 点)。未 def 先 use 的 VReg（如 graph 输入）start 记 0。
fn compute_live_intervals(instrs: &[Instruction]) -> Vec<LiveInterval> {
    let mut intervals: HashMap<VReg, LiveInterval> = HashMap::new();
    for (idx, ins) in instrs.iter().enumerate() {
        // output = def 点（start）
        for &vreg in &ins.outputs {
            intervals
                .entry(vreg)
                .and_modify(|li| {
                    // SSA 下每个 VReg 只 def 一次，但保险起见取 min/max
                    li.start = li.start.min(idx);
                    li.end = li.end.max(idx);
                })
                .or_insert(LiveInterval {
                    vreg,
                    start: idx,
                    end: idx,
                });
        }
        // input = use 点（end 扩展）
        for op in &ins.inputs {
            if let Operand::VReg(vreg) = op {
                intervals
                    .entry(*vreg)
                    .and_modify(|li| {
                        li.end = li.end.max(idx);
                    })
                    .or_insert(LiveInterval {
                        vreg: *vreg,
                        start: 0,
                        end: idx,
                    });
                // 注：SSA 下 VReg 应先 def 再 use，但保守处理 start=0
            }
        }
    }
    let mut result: Vec<_> = intervals.into_values().collect();
    result.sort_by_key(|li| li.start);
    result
}

/// 按 target 取物理寄存器数量上限
fn num_physical_regs(desc: &DeviceDesc) -> u32 {
    match desc {
        DeviceDesc::Cuda(c) => c.max_registers_per_thread,
        DeviceDesc::Npu(n) => n.vector_regs,
        DeviceDesc::Cpu => 16, // CPU 保守 16 个通用寄存器
    }
}

/// 线性扫描寄存器分配（Fast 模式，经典 Poletto-Sarkar 算法）。
///
/// 步骤：
/// 1. 按 start 排序所有 live interval（compute_live_intervals 已排）
/// 2. 维护 active list（已分配且未过期的 interval）+ free list（可用 PReg）
/// 3. 对每个 interval：
///    - 从 active list 移除已过期的（end < 当前 start），归还 PReg 到 free list
///    - 若 free list 有 PReg，分配之，加入 active
///    - 否则 spill（启发式：active 中 end 最大的比当前 interval 的 end 还大时，
///      spill 那个并让当前 interval 复用其 PReg；否则 spill 当前 interval）
fn linear_scan(instrs: &[Instruction], num_regs: u32) -> Result<Allocation> {
    let intervals = compute_live_intervals(instrs);
    let mut vreg_to_preg: HashMap<VReg, PReg> = HashMap::new();
    let mut spilled = Vec::new();

    // active list: (interval, PReg)，按 end 排序
    let mut active: Vec<(LiveInterval, PReg)> = Vec::new();
    // free list: 可用 PReg（初始全部空闲）
    let mut free_regs: Vec<PReg> = (0..num_regs).map(PReg).collect();

    for interval in intervals {
        // 过期处理：end < interval.start 的从 active 移除，归还 PReg
        let mut to_free: Vec<PReg> = Vec::new();
        active.retain(|(li, preg)| {
            if li.end < interval.start {
                to_free.push(*preg);
                false
            } else {
                true
            }
        });
        free_regs.extend(to_free);

        if let Some(preg) = free_regs.pop() {
            vreg_to_preg.insert(interval.vreg, preg);
            active.push((interval, preg));
            active.sort_by_key(|(li, _)| li.end);
        } else {
            // 无空闲寄存器，spill。启发式：若 active 中 end 最大的比当前 interval
            // 的 end 还大，spill 那个并把它的 PReg 给当前 interval（更短的优先留寄存器）
            if let Some((last_li, last_preg)) = active.last().copied() {
                if last_li.end > interval.end {
                    // spill active 中 end 最大的，当前 interval 复用其 PReg
                    vreg_to_preg.remove(&last_li.vreg);
                    spilled.push(last_li.vreg);
                    vreg_to_preg.insert(interval.vreg, last_preg);
                    active.pop();
                    active.push((interval, last_preg));
                    active.sort_by_key(|(li, _)| li.end);
                } else {
                    // 当前 interval end 更大或相等，spill 当前 interval
                    spilled.push(interval.vreg);
                }
            } else {
                // active 空但 free 也空（num_regs == 0），只能 spill 当前
                spilled.push(interval.vreg);
            }
        }
    }

    Ok(Allocation {
        vreg_to_preg,
        spilled,
    })
}

/// 寄存器分配入口
pub fn allocate(
    instrs: &[Instruction],
    arch_graph: &ArchGraph,
    mode: RegAllocMode,
) -> Result<Allocation> {
    let num_regs = num_physical_regs(&arch_graph.desc);
    match mode {
        RegAllocMode::Fast => linear_scan(instrs, num_regs),
        RegAllocMode::Standard | RegAllocMode::Quality => {
            // TODO: SSA 消φ + IRC，暂降级到 Fast
            linear_scan(instrs, num_regs)
        }
        RegAllocMode::Exhaustive => {
            // TODO: 暴力枚举，暂降级到 Fast
            linear_scan(instrs, num_regs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::Target;

    /// 构造测试用 Instruction：inputs 用 VReg，outputs 用 VReg
    fn make_instr(op: &str, inputs: Vec<u32>, outputs: Vec<u32>) -> Instruction {
        Instruction {
            op: op.to_string(),
            inputs: inputs.into_iter().map(VReg).map(Operand::VReg).collect(),
            outputs: outputs.into_iter().map(VReg).collect(),
        }
    }

    #[test]
    fn test_compute_live_intervals() {
        // add v0,v1 -> v2  (idx 0: def v2, use v0 v1)
        // sub v2,v3 -> v4   (idx 1: def v4, use v2 v3)
        // mul v4,v0 -> v5   (idx 2: def v5, use v4 v0)
        let instrs = vec![
            make_instr("add", vec![0, 1], vec![2]),
            make_instr("sub", vec![2, 3], vec![4]),
            make_instr("mul", vec![4, 0], vec![5]),
        ];
        let li = compute_live_intervals(&instrs);
        let by_vreg: HashMap<VReg, LiveInterval> = li.into_iter().map(|x| (x.vreg, x)).collect();
        // v0: 仅 use（idx 0 和 2），start=0（保守），end=2
        let v0 = by_vreg.get(&VReg(0)).expect("v0 应有 interval");
        assert_eq!(v0.start, 0);
        assert_eq!(v0.end, 2, "v0 最后 use 在 idx 2");
        // v2: def@0, use@1
        let v2 = by_vreg.get(&VReg(2)).expect("v2 应有 interval");
        assert_eq!(v2.start, 0);
        assert_eq!(v2.end, 1, "v2 start=0 end=1");
        // v5: def@2, 不再 use
        let v5 = by_vreg.get(&VReg(5)).expect("v5 应有 interval");
        assert_eq!(v5.start, 2);
        assert_eq!(v5.end, 2, "v5 start=2 end=2");
    }

    #[test]
    fn test_linear_scan_no_spill() {
        // num_regs=4，构造的 VReg 都不重叠，全部应映射到 PReg 0-3，spilled 空
        // v0 = const -> v0   (idx 0, [0,0])
        // v1 = const -> v1   (idx 1, [1,1])
        // v2 = add v0,v1 -> v2 (idx 2, use v0 v1, def v2, [0,2])
        // v3 = add v2,imm -> v3 (idx 3, def v3 [3,3])
        let instrs = vec![
            make_instr("const", vec![], vec![0]),
            make_instr("const", vec![], vec![1]),
            make_instr("add", vec![0, 1], vec![2]),
            make_instr("add", vec![2], vec![3]),
        ];
        let alloc = linear_scan(&instrs, 4).unwrap();
        assert!(alloc.spilled.is_empty(), "无重叠不应 spill");
        // 4 个 VReg 都应分配到 PReg 0-3
        assert_eq!(alloc.vreg_to_preg.len(), 4);
        for v in [0, 1, 2, 3] {
            let preg = alloc
                .vreg_to_preg
                .get(&VReg(v))
                .unwrap_or_else(|| panic!("VReg({}) 应已分配", v));
            assert!(preg.0 < 4, "PReg 应在 0-3 范围内");
        }
    }

    #[test]
    fn test_linear_scan_with_spill() {
        // num_regs=1，两个重叠的 VReg，应 spill 一个
        // v0 def@0 use@2，v1 def@1 use@2——两者在 idx 2 同时活跃，只有一个寄存器
        let instrs = vec![
            make_instr("const", vec![], vec![0]),   // v0 def@0
            make_instr("const", vec![], vec![1]),   // v1 def@1
            make_instr("add", vec![0, 1], vec![2]), // idx 2 同时 use v0 v1
        ];
        let alloc = linear_scan(&instrs, 1).unwrap();
        assert!(
            !alloc.spilled.is_empty(),
            "num_regs=1 且有重叠，应至少 spill 一个"
        );
    }

    #[test]
    fn test_allocate_dispatches_by_mode() {
        // ArchGraph::new(Cuda) desc 是 CudaDesc，Fast 模式，空指令不应报错
        let ag = ArchGraph::new(Target::Cuda);
        let alloc = allocate(&[], &ag, RegAllocMode::Fast).unwrap();
        assert!(alloc.vreg_to_preg.is_empty());
        assert!(alloc.spilled.is_empty());
        // Standard/Quality/Exhaustive 暂降级 Fast，也应正常返回
        let _ = allocate(&[], &ag, RegAllocMode::Standard).unwrap();
        let _ = allocate(&[], &ag, RegAllocMode::Quality).unwrap();
        let _ = allocate(&[], &ag, RegAllocMode::Exhaustive).unwrap();
    }

    #[test]
    fn test_num_physical_regs_per_target() {
        // Cuda=255、Npu=256、Cpu=16
        assert_eq!(
            num_physical_regs(&DeviceDesc::Cuda(arch::cuda::CudaDesc::default())),
            255
        );
        assert_eq!(
            num_physical_regs(&DeviceDesc::Npu(arch::npu::NpuDesc::default())),
            256
        );
        assert_eq!(num_physical_regs(&DeviceDesc::Cpu), 16);
        // 通过 ArchGraph::new 验证 desc 初始化正确
        assert_eq!(num_physical_regs(&ArchGraph::new(Target::Cuda).desc), 255);
        assert_eq!(num_physical_regs(&ArchGraph::new(Target::Npu).desc), 256);
        assert_eq!(num_physical_regs(&ArchGraph::new(Target::Cpu).desc), 16);
    }
}
