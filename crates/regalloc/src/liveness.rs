//! liveness — 活跃区间分析
//!
//! 扫描 MachineInstr 序列，计算每个 VReg 的活跃区间 [def, last_use]。
//! 活跃区间是寄存器分配的核心数据：如果两个 VReg 的活跃区间重叠，
//! 它们不能共用同一个物理寄存器（会产生干扰）。
//!
//! 算法：
//! 1. 对每条指令，记录该指令中每个 VReg 的 def（定义）和 use（使用）
//! 2. VReg 的活跃区间 = [首次 def 位置, 最后一次 use 位置]
//! 3. 如果 VReg 被 use 但从未 def（图输入），区间从 0 开始
//! 4. 如果 VReg 被 def 但从未 use（死代码），区间 = [def, def]（单点）

use crate::types::*;
use std::collections::HashMap;

/// 活跃区间 [start, end]（指令序号，闭区间）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveInterval {
    pub start: u32,
    pub end: u32,
}

impl LiveInterval {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// 检查两个活跃区间是否重叠（干扰）
    /// 重叠条件：区间 A 的结束点 >= B 的开始点，且 B 的结束点 >= A 的开始点
    /// 使用 >= 而非 > 是因为：如果 v0 在指令 i 被 use，v1 在指令 i 被 def，
    /// 它们在指令 i 同时活跃，必须干扰（不能共用寄存器）。
    /// 例外情况（move 指令的源和目标可共用）由 coalescing 阶段处理。
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        self.end >= other.start && other.end >= self.start
    }

    pub fn len(&self) -> u32 {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            0
        }
    }

    /// 区间是否为空（end < start 视为空，正常情况下不会出现）
    pub fn is_empty(&self) -> bool {
        self.end < self.start
    }
}

/// 活跃分析结果
#[derive(Debug, Clone, Default)]
pub struct LivenessResult {
    /// VReg → 活跃区间
    pub intervals: HashMap<VReg, LiveInterval>,
    /// 每个 VReg 的使用次数（用于溢出代价计算）
    pub use_counts: HashMap<VReg, usize>,
}

impl LivenessResult {
    pub fn get(&self, vreg: VReg) -> Option<&LiveInterval> {
        self.intervals.get(&vreg)
    }

    pub fn use_count(&self, vreg: VReg) -> usize {
        self.use_counts.get(&vreg).copied().unwrap_or(0)
    }
}

/// 对 MachineInstr 序列执行活跃分析
///
/// 返回每个 VReg 的活跃区间和使用次数。
pub fn analyze(instructions: &[MachineInstr]) -> LivenessResult {
    let mut intervals: HashMap<VReg, LiveInterval> = HashMap::new();
    let mut use_counts: HashMap<VReg, usize> = HashMap::new();

    for (idx, instr) in instructions.iter().enumerate() {
        let pos = idx as u32;

        // 处理 defs（输出）：更新或创建区间
        for vreg in instr.vreg_defs() {
            intervals
                .entry(vreg)
                .and_modify(|iv| {
                    // 如果已有区间（多个 def），取最早的 start
                    if pos < iv.start {
                        iv.start = pos;
                    }
                    // end 至少到 def 位置
                    if pos > iv.end {
                        iv.end = pos;
                    }
                })
                .or_insert(LiveInterval::new(pos, pos));
        }

        // 处理 uses（输入）：更新区间 end
        for vreg in instr.vreg_uses() {
            *use_counts.entry(vreg).or_insert(0) += 1;
            intervals
                .entry(vreg)
                .and_modify(|iv| {
                    // use 延长区间 end
                    if pos > iv.end {
                        iv.end = pos;
                    }
                })
                .or_insert_with(|| {
                    // use 但未 def（图输入值）：区间从 0 开始
                    LiveInterval::new(0, pos)
                });
        }
    }

    LivenessResult {
        intervals,
        use_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_instr(op: &str, uses: Vec<u32>, defs: Vec<u32>) -> MachineInstr {
        MachineInstr {
            op: op.to_string(),
            operands: uses.into_iter().map(VReg).map(Operand::VReg).collect(),
            defs: defs.into_iter().map(VReg).map(Operand::VReg).collect(),
            args: vec![],
        }
    }

    #[test]
    fn interval_overlaps_basic() {
        let a = LiveInterval::new(0, 3);
        let b = LiveInterval::new(2, 5);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn interval_overlap_at_same_instruction() {
        // A 在 3 结束，B 在 3 开始 → 重叠（同一指令中 v0 被 use、v1 被 def）
        let a = LiveInterval::new(0, 3);
        let b = LiveInterval::new(3, 5);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn interval_no_overlap_separate() {
        // A [0,1] B [3,4] → 不重叠（中间有间隔）
        let a = LiveInterval::new(0, 1);
        let b = LiveInterval::new(3, 4);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn analyze_simple_chain() {
        // v0 = load   (def v0 @ 0)
        // v1 = fadd v0 v0  (use v0 @ 1, def v1 @ 1)
        // v2 = fmul v1 v0  (use v1,v0 @ 2, def v2 @ 2)
        // store v2         (use v2 @ 3)
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("fadd", vec![0, 0], vec![1]),
            mk_instr("fmul", vec![1, 0], vec![2]),
            mk_instr("store", vec![2], vec![]),
        ];

        let result = analyze(&instrs);

        // v0: [0, 2] — def @ 0, last use @ 2
        let v0 = result.get(VReg(0)).unwrap();
        assert_eq!(v0.start, 0);
        assert_eq!(v0.end, 2);

        // v1: [1, 2] — def @ 1, last use @ 2
        let v1 = result.get(VReg(1)).unwrap();
        assert_eq!(v1.start, 1);
        assert_eq!(v1.end, 2);

        // v2: [2, 3] — def @ 2, last use @ 3
        let v2 = result.get(VReg(2)).unwrap();
        assert_eq!(v2.start, 2);
        assert_eq!(v2.end, 3);

        // use counts
        assert_eq!(result.use_count(VReg(0)), 3); // 2 in fadd + 1 in fmul
        assert_eq!(result.use_count(VReg(1)), 1);
        assert_eq!(result.use_count(VReg(2)), 1);
    }

    #[test]
    fn analyze_graph_input() {
        // v0 是图输入（被 use 但从未 def）
        // fadd v0 → v1  (use v0 @ 0, def v1 @ 0)
        // store v1       (use v1 @ 1)
        let instrs = vec![
            mk_instr("fadd", vec![0], vec![1]),
            mk_instr("store", vec![1], vec![]),
        ];

        let result = analyze(&instrs);

        // v0: 未 def，use @ 0 → 区间 [0, 0]
        let v0 = result.get(VReg(0)).unwrap();
        assert_eq!(v0.start, 0);
        assert_eq!(v0.end, 0);

        // v1: def @ 0, use @ 1 → 区间 [0, 1]
        let v1 = result.get(VReg(1)).unwrap();
        assert_eq!(v1.start, 0);
        assert_eq!(v1.end, 1);
    }

    #[test]
    fn analyze_dead_code() {
        // v0 被 def 但从未 use（死代码）
        // v0 = load (def v0 @ 0)
        // v1 = load (def v1 @ 1)
        // store v1  (use v1 @ 2)
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("load", vec![], vec![1]),
            mk_instr("store", vec![1], vec![]),
        ];

        let result = analyze(&instrs);

        // v0: def @ 0, never used → [0, 0]（单点）
        let v0 = result.get(VReg(0)).unwrap();
        assert_eq!(v0.start, 0);
        assert_eq!(v0.end, 0);

        // v0 use count = 0
        assert_eq!(result.use_count(VReg(0)), 0);
    }

    #[test]
    fn intervals_determine_interference() {
        // v0 [0,2] 和 v1 [1,3] 重叠 → 干扰
        // v0 [0,2] 和 v2 [2,3] 在指令 2 同时活跃（v0 被 use，v2 被 def）→ 干扰
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),     // v0 def @ 0
            mk_instr("load", vec![], vec![1]),     // v1 def @ 1
            mk_instr("fadd", vec![0, 1], vec![2]), // v0,v1 use @ 2, v2 def @ 2
            mk_instr("fmul", vec![1, 2], vec![3]), // v1,v2 use @ 3, v3 def @ 3
            mk_instr("store", vec![3], vec![]),    // v3 use @ 4
        ];

        let result = analyze(&instrs);

        let v0 = result.get(VReg(0)).unwrap(); // [0, 2]
        let v1 = result.get(VReg(1)).unwrap(); // [1, 3]
        let v2 = result.get(VReg(2)).unwrap(); // [2, 3]

        // v0 和 v1 重叠 → 干扰
        assert!(v0.overlaps(v1));
        // v0 和 v2 在指令 2 同时活跃 → 干扰
        assert!(v0.overlaps(v2));
    }
}
