//! types — 寄存器分配器的核心类型定义
//!
//! VReg：虚拟寄存器（IR 值的抽象表示）
//! PReg：物理寄存器（目标架构的实际寄存器）
//! Operand：指令操作数（VReg 或 PReg）
//! MachineInstr：机器指令（带 VReg operand 的中间表示）
//! RegisterFile：物理寄存器文件描述（按目标架构区分）

use std::fmt;

/// 虚拟寄存器
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(pub u32);

/// 物理寄存器
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(pub u32);

impl VReg {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl PReg {
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%v{}", self.0)
    }
}

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "R{}", self.0)
    }
}

/// 指令操作数
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    /// 虚拟寄存器（分配前）
    VReg(VReg),
    /// 物理寄存器（分配后）
    PReg(PReg),
}

impl Operand {
    pub fn as_vreg(&self) -> Option<VReg> {
        match self {
            Operand::VReg(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_preg(&self) -> Option<PReg> {
        match self {
            Operand::PReg(p) => Some(*p),
            _ => None,
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::VReg(v) => write!(f, "{}", v),
            Operand::PReg(p) => write!(f, "{}", p),
        }
    }
}

/// 机器指令（寄存器分配的中间表示）
#[derive(Debug, Clone)]
pub struct MachineInstr {
    /// 操作名（如 "fadd"、"mma"、"relu"）
    pub op: String,
    /// 输入操作数（VReg，分配后替换为 PReg）
    pub operands: Vec<Operand>,
    /// 输出操作数（VReg，分配后替换为 PReg）
    pub defs: Vec<Operand>,
    /// 立即数参数（如 "axis"、imm 值等，不参与寄存器分配）
    pub args: Vec<String>,
}

impl MachineInstr {
    /// 获取所有使用的 VReg（operands 中的）
    pub fn vreg_uses(&self) -> Vec<VReg> {
        self.operands.iter().filter_map(|o| o.as_vreg()).collect()
    }

    /// 获取所有定义的 VReg（defs 中的）
    pub fn vreg_defs(&self) -> Vec<VReg> {
        self.defs.iter().filter_map(|o| o.as_vreg()).collect()
    }

    /// 是否是 move 指令（op == "mov"，只有一个 operand 和一个 def）
    pub fn is_move(&self) -> bool {
        self.op == "mov" && self.operands.len() == 1 && self.defs.len() == 1
    }

    /// 获取 move 的源 VReg
    pub fn move_src(&self) -> Option<VReg> {
        if self.is_move() {
            self.operands[0].as_vreg()
        } else {
            None
        }
    }

    /// 获取 move 的目标 VReg
    pub fn move_dst(&self) -> Option<VReg> {
        if self.is_move() {
            self.defs[0].as_vreg()
        } else {
            None
        }
    }

    /// 格式化输出（调试用）
    pub fn display(&self) -> String {
        let ops: Vec<String> = self.operands.iter().map(|o| o.to_string()).collect();
        let defs: Vec<String> = self.defs.iter().map(|o| o.to_string()).collect();
        let args = if self.args.is_empty() {
            String::new()
        } else {
            format!(" {}", self.args.join(" "))
        };
        format!(
            "{} {} -> {}{}",
            self.op,
            ops.join(", "),
            defs.join(", "),
            args
        )
    }
}

/// 寄存器分配结果
#[derive(Debug, Clone)]
pub struct Allocation {
    /// 分配后的指令序列（VReg 已替换为 PReg，溢出的插入了 load/store）
    pub instructions: Vec<MachineInstr>,
    /// VReg → PReg 映射
    pub vreg_to_preg: std::collections::HashMap<VReg, PReg>,
    /// 溢出到栈的 VReg 集合
    pub spilled: std::collections::HashSet<VReg>,
    /// 溢出槽数量
    pub spill_slots: usize,
}

/// 物理寄存器文件描述
#[derive(Debug, Clone)]
pub struct RegisterFile {
    /// 可用物理寄存器数量
    pub num_registers: u32,
    /// 保留寄存器（如栈指针等，不参与分配）
    pub reserved: Vec<PReg>,
}

impl RegisterFile {
    /// CUDA 架构寄存器文件（32 个物理寄存器，保留 2 个）
    pub fn cuda() -> Self {
        Self {
            num_registers: 32,
            reserved: vec![PReg(30), PReg(31)], // 栈指针 + 返回地址
        }
    }

    /// NPU 架构寄存器文件
    pub fn npu() -> Self {
        Self {
            num_registers: 32,
            reserved: vec![PReg(31)],
        }
    }

    /// CPU 架构寄存器文件（16 个，保留 3 个）
    pub fn cpu() -> Self {
        Self {
            num_registers: 16,
            reserved: vec![PReg(13), PReg(14), PReg(15)], // SP / BP / RA
        }
    }

    /// 按目标架构选择
    pub fn for_target(target: common::Target) -> Self {
        match target {
            common::Target::Cuda => Self::cuda(),
            common::Target::Npu => Self::npu(),
            common::Target::Cpu => Self::cpu(),
        }
    }

    /// 可分配的物理寄存器列表
    pub fn allocatable(&self) -> Vec<PReg> {
        let reserved_set: std::collections::HashSet<PReg> = self.reserved.iter().copied().collect();
        (0..self.num_registers)
            .map(PReg)
            .filter(|p| !reserved_set.contains(p))
            .collect()
    }

    /// 可分配寄存器数量（K）
    pub fn k(&self) -> usize {
        self.allocatable().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vreg_display() {
        assert_eq!(VReg(0).to_string(), "%v0");
        assert_eq!(VReg(42).to_string(), "%v42");
    }

    #[test]
    fn preg_display() {
        assert_eq!(PReg(0).to_string(), "R0");
        assert_eq!(PReg(15).to_string(), "R15");
    }

    #[test]
    fn operand_as_vreg() {
        let v = Operand::VReg(VReg(5));
        assert_eq!(v.as_vreg(), Some(VReg(5)));
        assert_eq!(v.as_preg(), None);

        let p = Operand::PReg(PReg(3));
        assert_eq!(p.as_preg(), Some(PReg(3)));
        assert_eq!(p.as_vreg(), None);
    }

    #[test]
    fn machine_instr_vreg_uses_defs() {
        let instr = MachineInstr {
            op: "fadd".into(),
            operands: vec![Operand::VReg(VReg(0)), Operand::VReg(VReg(1))],
            defs: vec![Operand::VReg(VReg(2))],
            args: vec![],
        };
        assert_eq!(instr.vreg_uses(), vec![VReg(0), VReg(1)]);
        assert_eq!(instr.vreg_defs(), vec![VReg(2)]);
        assert!(!instr.is_move());
    }

    #[test]
    fn machine_instr_move_detection() {
        let mov = MachineInstr {
            op: "mov".into(),
            operands: vec![Operand::VReg(VReg(0))],
            defs: vec![Operand::VReg(VReg(1))],
            args: vec![],
        };
        assert!(mov.is_move());
        assert_eq!(mov.move_src(), Some(VReg(0)));
        assert_eq!(mov.move_dst(), Some(VReg(1)));
    }

    #[test]
    fn register_file_allocatable() {
        let rf = RegisterFile::cpu();
        let alloc = rf.allocatable();
        // 16 - 3 reserved = 13
        assert_eq!(alloc.len(), 13);
        assert!(!alloc.contains(&PReg(13)));
        assert!(!alloc.contains(&PReg(14)));
        assert!(!alloc.contains(&PReg(15)));
        assert!(alloc.contains(&PReg(0)));
    }

    #[test]
    fn register_file_cuda() {
        let rf = RegisterFile::cuda();
        let alloc = rf.allocatable();
        // 32 - 2 reserved = 30
        assert_eq!(alloc.len(), 30);
        assert!(!alloc.contains(&PReg(30)));
        assert!(!alloc.contains(&PReg(31)));
    }

    #[test]
    fn register_file_for_target() {
        let rf = RegisterFile::for_target(common::Target::Cuda);
        assert_eq!(rf.k(), 30);
        let rf = RegisterFile::for_target(common::Target::Cpu);
        assert_eq!(rf.k(), 13);
    }
}
