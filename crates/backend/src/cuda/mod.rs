//! cuda — CUDA C++ 后端代码生成
//!
//! 生成 `.cu` kernel 源码，支持微架构：
//! - Ampere SM80：`mma.sync` + `cp.async` + `__ldg`
//! - Hopper SM90：`wgmma.mma_async` + TMA (Tensor Memory Accelerator) + `cp.async.bulk`
//! - Blackwell SM100：tensor memory + FP4/FP6 原生 + `wgmma` 扩展

pub mod emit;
pub mod kernels;

pub use emit::emit;
