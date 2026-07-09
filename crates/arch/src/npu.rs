//! npu — Ascend NPU 后端描述

/// Ascend NPU 架构描述
#[derive(Debug, Clone, Copy)]
pub struct NpuDesc {
    pub soc_version: NpuSoc,
    pub aicore_count: u32,
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
        }
    }
}
