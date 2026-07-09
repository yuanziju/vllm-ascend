//! cuda — CUDA 后端描述

/// CUDA 架构描述
#[derive(Debug, Clone, Copy)]
pub struct CudaDesc {
    pub compute_capability: (u32, u32),
    pub shared_mem_bytes: usize,
}

impl Default for CudaDesc {
    fn default() -> Self {
        Self {
            compute_capability: (8, 0),
            shared_mem_bytes: 48 * 1024,
        }
    }
}
