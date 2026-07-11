//! cuda — CUDA 后端描述

/// CUDA 架构描述（含 register file 参数，供寄存器分配读）
#[derive(Debug, Clone, Copy)]
pub struct CudaDesc {
    pub compute_capability: (u32, u32),
    pub shared_mem_bytes: usize,
    /// 每 SM 寄存器总数（Ampere = 65536）
    pub registers_per_sm: u32,
    /// 每线程最大寄存器数（Ampere = 255）
    pub max_registers_per_thread: u32,
    /// warp 大小
    pub warp_size: u32,
}

impl Default for CudaDesc {
    fn default() -> Self {
        Self {
            compute_capability: (8, 0),
            shared_mem_bytes: 48 * 1024,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            warp_size: 32,
        }
    }
}
