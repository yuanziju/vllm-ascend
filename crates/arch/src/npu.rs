//! npu — Ascend NPU 后端描述

/// Ascend NPU 架构描述（含寄存器/UB 参数，供寄存器分配读）
#[derive(Debug, Clone, Copy)]
pub struct NpuDesc {
    pub soc_version: NpuSoc,
    pub aicore_count: u32,
    /// Vector 寄存器数量（Ascend 估算，不同 SoC 略有差异）
    pub vector_regs: u32,
    /// Unified Buffer 大小（字节，Ascend910 约 256KB）
    pub ub_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuSoc {
    Ascend910B1,
    Ascend910B3,
    Ascend310P3,
}

impl Default for NpuDesc {
    fn default() -> Self {
        Self {
            soc_version: NpuSoc::Ascend910B1,
            aicore_count: 32,
            vector_regs: 256,
            ub_bytes: 256 * 1024,
        }
    }
}
