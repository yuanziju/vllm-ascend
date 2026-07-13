//! allocator — 寄存器分配入口
//!
//! 串联完整的寄存器分配流程：
//! 1. 活跃分析 → 2. 干扰图构建 → 3. 保守合并 → 4. 图着色 → 5. 溢出重写
//!
//! 溢出重写：被溢出的 VReg 在 def 点后插 store 指令，
//! 在 use 点前插 load 指令（分段溢出，非全区间）。

use crate::coalescing::{coalesce, find_coalesce_pairs};
use crate::coloring::color;
use crate::interference::build;
use crate::liveness::analyze;
use crate::types::*;
use std::collections::{HashMap, HashSet};

/// 寄存器分配入口
///
/// 参数：
/// - instructions: VReg 形式的 MachineInstr 序列
/// - reg_file: 物理寄存器文件
///
/// 返回分配结果（PReg 形式的指令 + 映射 + 溢出信息）
pub fn allocate(instructions: &[MachineInstr], reg_file: &RegisterFile) -> Allocation {
    // 1. 活跃分析
    let liveness = analyze(instructions);

    // 2. 干扰图构建（含 move 例外，便于后续 coalescing）
    let mut graph = build(&liveness, instructions);

    // 3. 保守合并（Briggs，阈值=3）
    let coalesce_pairs = find_coalesce_pairs(instructions);
    let coalesced = coalesce(&mut graph, &coalesce_pairs, reg_file.k(), 3);

    // 构建合并映射：被合并的 VReg → 保留的 VReg
    let mut coalesce_map: HashMap<VReg, VReg> = HashMap::new();
    for pair in &coalesced {
        coalesce_map.insert(pair.dst, pair.src);
    }

    // 4. 图着色
    let colors = reg_file.allocatable();
    let coloring = color(&graph, &liveness, &colors);

    // 5. 溢出重写 + PReg 替换
    let mut spill_slots = 0usize;
    let mut spill_map: HashMap<VReg, usize> = HashMap::new();

    // 为每个溢出的 VReg 分配一个栈槽
    for &vreg in &coloring.spilled {
        spill_map.insert(vreg, spill_slots);
        spill_slots += 1;
    }

    // 重写指令
    let mut result_instrs = Vec::new();
    let coalesced_set: HashSet<VReg> = coalesced.iter().map(|p| p.dst).collect();

    for instr in instructions {
        // 跳过被合并的 move 指令（源已与目标合并，move 冗余）
        if instr.is_move() {
            if let (Some(_src), Some(dst)) = (instr.move_src(), instr.move_dst()) {
                if coalesced_set.contains(&dst) {
                    continue; // 跳过冗余 move
                }
            }
        }

        let mut new_instr = MachineInstr {
            op: instr.op.clone(),
            operands: Vec::new(),
            defs: Vec::new(),
            args: instr.args.clone(),
        };

        // 同指令内已 load 过的 VReg → 复用同一临时寄存器（避免重复 load_spill）
        // 例如 `fadd v0 v0`（v0 被溢出）只应插一条 load_spill，两个 operand 都用 R0
        let mut loaded_in_this_instr: HashMap<VReg, PReg> = HashMap::new();

        // 处理 operands（use）：溢出的 VReg 前插 load
        for &operand in &instr.operands {
            if let Operand::VReg(vreg) = operand {
                // 解析合并映射
                let effective_vreg = coalesce_map.get(&vreg).copied().unwrap_or(vreg);

                if coloring.spilled.contains(&effective_vreg) {
                    if let Some(&preg) = loaded_in_this_instr.get(&effective_vreg) {
                        // 同指令内已 load 过此 VReg，直接复用临时寄存器
                        new_instr.operands.push(Operand::PReg(preg));
                    } else {
                        let slot = spill_map[&effective_vreg];
                        let preg = PReg(0); // 用 R0 作临时寄存器
                                            // 插入 load 指令：从栈槽加载到临时寄存器
                        result_instrs.push(MachineInstr {
                            op: "load_spill".into(),
                            operands: vec![],
                            defs: vec![Operand::PReg(preg)],
                            args: vec![format!("spill{}", slot)],
                        });
                        new_instr.operands.push(Operand::PReg(preg));
                        loaded_in_this_instr.insert(effective_vreg, preg);
                    }
                } else if let Some(&preg) = coloring.assignment.get(&effective_vreg) {
                    new_instr.operands.push(Operand::PReg(preg));
                } else {
                    // 未分配也未溢出的 VReg（可能是图输入未使用）
                    new_instr.operands.push(Operand::VReg(effective_vreg));
                }
            } else {
                new_instr.operands.push(operand);
            }
        }

        // 处理 defs（输出）：溢出的 VReg 后插 store
        for &operand in &instr.defs {
            if let Operand::VReg(vreg) = operand {
                let effective_vreg = coalesce_map.get(&vreg).copied().unwrap_or(vreg);

                if coloring.spilled.contains(&effective_vreg) {
                    let slot = spill_map[&effective_vreg];
                    // 输出到临时寄存器，然后 store 到栈槽
                    new_instr.defs.push(Operand::PReg(PReg(0)));
                    result_instrs.push(new_instr.clone());

                    result_instrs.push(MachineInstr {
                        op: "store_spill".into(),
                        operands: vec![Operand::PReg(PReg(0))],
                        defs: vec![],
                        args: vec![format!("spill{}", slot)],
                    });

                    // new_instr 已用完，创建新的空指令给后续 def
                    new_instr = MachineInstr {
                        op: instr.op.clone(),
                        operands: new_instr.operands.clone(),
                        defs: Vec::new(),
                        args: instr.args.clone(),
                    };
                } else if let Some(&preg) = coloring.assignment.get(&effective_vreg) {
                    new_instr.defs.push(Operand::PReg(preg));
                } else {
                    new_instr.defs.push(Operand::VReg(effective_vreg));
                }
            } else {
                new_instr.defs.push(operand);
            }
        }

        // 如果指令没有被溢出 def 提前 push，则添加
        if !new_instr.defs.is_empty() || !new_instr.operands.is_empty() || new_instr.op != instr.op
        {
            result_instrs.push(new_instr);
        }
    }

    // 构建 vreg_to_preg 映射（含合并）
    let mut vreg_to_preg: HashMap<VReg, PReg> = HashMap::new();
    for (&vreg, &preg) in &coloring.assignment {
        vreg_to_preg.insert(vreg, preg);
        // 合并的 VReg 也映射到相同的 PReg
        for pair in &coalesced {
            if pair.src == vreg {
                vreg_to_preg.insert(pair.dst, preg);
            }
        }
    }

    Allocation {
        instructions: result_instrs,
        vreg_to_preg,
        spilled: coloring.spilled,
        spill_slots,
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
    fn allocate_simple_chain() {
        // v0 = load
        // v1 = fadd v0
        // v2 = fmul v1
        // store v2
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("fadd", vec![0], vec![1]),
            mk_instr("fmul", vec![1], vec![2]),
            mk_instr("store", vec![2], vec![]),
        ];

        let rf = RegisterFile::cpu(); // 13 registers
        let result = allocate(&instrs, &rf);

        // 不应溢出
        assert!(result.spilled.is_empty());
        // 所有 VReg 应被分配
        assert_eq!(result.vreg_to_preg.len(), 3);
        // v0 和 v1 干扰，v1 和 v2 干扰，但 v0 和 v2 不干扰 → 可共用
        let p0 = result.vreg_to_preg[&VReg(0)];
        let p1 = result.vreg_to_preg[&VReg(1)];
        let p2 = result.vreg_to_preg[&VReg(2)];
        assert_ne!(p0, p1); // v0-v1 干扰
        assert_ne!(p1, p2); // v1-v2 干扰
        assert_eq!(p0, p2); // v0-v2 不干扰 → 可共用
    }

    #[test]
    fn allocate_with_spill() {
        // 5 个互相干扰的 VReg，只有 2 个寄存器 → 必须溢出
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("load", vec![], vec![1]),
            mk_instr("load", vec![], vec![2]),
            mk_instr("load", vec![], vec![3]),
            mk_instr("load", vec![], vec![4]),
            mk_instr("use", vec![0, 1, 2, 3, 4], vec![]),
        ];

        // 只用 2 个寄存器
        let rf = RegisterFile {
            num_registers: 2,
            reserved: vec![],
        };
        let result = allocate(&instrs, &rf);

        // 应有溢出
        assert!(!result.spilled.is_empty());
        assert!(result.spill_slots > 0);

        // 结果中应有 load_spill / store_spill 指令
        let has_load_spill = result.instructions.iter().any(|i| i.op == "load_spill");
        let has_store_spill = result.instructions.iter().any(|i| i.op == "store_spill");
        assert!(has_load_spill || has_store_spill);
    }

    #[test]
    fn allocate_empty() {
        let rf = RegisterFile::cpu();
        let result = allocate(&[], &rf);
        assert!(result.instructions.is_empty());
        assert!(result.vreg_to_preg.is_empty());
    }

    #[test]
    fn allocate_coalesces_moves() {
        // v0 = load
        // v1 = mov v0  (move 指令)
        // v2 = fadd v1
        // store v2
        // v0 和 v1 不干扰（v0 在 mov 后不再活跃）→ 可合并
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("mov", vec![0], vec![1]),
            mk_instr("fadd", vec![1], vec![2]),
            mk_instr("store", vec![2], vec![]),
        ];

        let rf = RegisterFile::cpu();
        let result = allocate(&instrs, &rf);

        // v0 和 v1 应合并为同一物理寄存器
        if let (Some(&p0), Some(&p1)) = (
            result.vreg_to_preg.get(&VReg(0)),
            result.vreg_to_preg.get(&VReg(1)),
        ) {
            assert_eq!(p0, p1, "v0 和 v1 应合并为同一物理寄存器");
        }

        // move 指令应被消除
        let has_mov = result.instructions.iter().any(|i| i.op == "mov");
        assert!(!has_mov, "move 指令应被合并消除");
    }

    #[test]
    fn allocate_many_non_interfering_share_register() {
        // 10 个不互相干扰的 VReg → 全部分配同一个寄存器
        let mut instrs = Vec::new();
        for i in 0..10u32 {
            instrs.push(mk_instr("load", vec![], vec![i]));
            instrs.push(mk_instr("store", vec![i], vec![]));
        }

        let rf = RegisterFile {
            num_registers: 1,
            reserved: vec![],
        };
        let result = allocate(&instrs, &rf);

        // 不应溢出
        assert!(result.spilled.is_empty());
        // 所有 VReg 应分配到 R0
        for i in 0..10u32 {
            assert_eq!(result.vreg_to_preg[&VReg(i)], PReg(0));
        }
    }

    /// 同一条指令中两次 use 同一个被溢出的 VReg（如 `fadd v0 v0`）时，
    /// 只应插入一条 load_spill，且两个 operand 都引用同一个临时寄存器。
    /// 当前实现每条 use 独立插 load，会产生冗余 load_spill —— 此测试期望优化后的行为。
    #[test]
    fn allocate_same_instr_reuses_spill_load() {
        // 构造：v0 区间最长（cost 最低 → 优先溢出），且 v0 在 use 指令里被用两次
        //   load v0          # v0 def @ 0   区间 [0,7]
        //   nop              # @ 1
        //   nop              # @ 2
        //   nop              # @ 3
        //   load v1          # v1 def @ 4   区间 [4,7]
        //   load v2          # v2 def @ 5   区间 [5,7]
        //   load v3          # v3 def @ 6   区间 [6,7]
        //   use v0 v0 v1 v2 v3   # @ 7  v0 被用两次
        //   （v0/v1/v2/v3 都在 @7 活跃 → 4-clique 互干扰）
        // K=2 时 4 个互干扰 VReg 至少要溢出 2 个；
        // spill_cost：v0=3/8≈0.38 最低 → v0 必被溢出
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("nop", vec![], vec![]),
            mk_instr("nop", vec![], vec![]),
            mk_instr("nop", vec![], vec![]),
            mk_instr("load", vec![], vec![1]),
            mk_instr("load", vec![], vec![2]),
            mk_instr("load", vec![], vec![3]),
            mk_instr("use", vec![0, 0, 1, 2, 3], vec![]), // v0 被用两次
        ];

        // 只用 2 个寄存器 → 强制溢出
        let rf = RegisterFile {
            num_registers: 2,
            reserved: vec![],
        };
        let result = allocate(&instrs, &rf);

        // v0 应被溢出（cost 最低）
        assert!(
            result.spilled.contains(&VReg(0)),
            "v0 应被溢出（cost 最低），实际 spilled = {:?}",
            result.spilled
        );

        // 找到 use 指令（结果中应只有一条 op=use）
        let use_instrs: Vec<_> = result
            .instructions
            .iter()
            .filter(|i| i.op == "use")
            .collect();
        assert_eq!(use_instrs.len(), 1, "应只有一条 use 指令");

        // 在 use 之前紧邻的 load_spill 应只有两条：
        //   1 条 for v0（v0 在 use 中被用两次，应共享同一条 load_spill）
        //   1 条 for v1（也被溢出）
        // 当前未优化的实现会产生 3 条（v0 重复 use 各插一条 load_spill）
        let use_pos = result
            .instructions
            .iter()
            .position(|i| i.op == "use")
            .expect("应找到 use");
        let load_spill_before_use: Vec<_> = result.instructions[..use_pos]
            .iter()
            .rev()
            .take_while(|i| i.op == "load_spill")
            .collect();
        assert_eq!(
            load_spill_before_use.len(),
            2,
            "use 前应只有 2 条 load_spill（v0 共享 1 条 + v1 1 条），实际有 {} 条\n完整序列: {:?}",
            load_spill_before_use.len(),
            result
                .instructions
                .iter()
                .map(|i| i.op.clone())
                .collect::<Vec<_>>()
        );
    }
}
