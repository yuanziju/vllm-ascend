//! triton — Triton Python DSL 后端代码生成
//!
//! 生成 `@triton.jit` 装饰的 Python kernel 源码。
//! Triton 自动处理寄存器分配和共享内存，但 kernel 逻辑仍需按算子生成。
//! 支持 SM90 TMA descriptor。

pub mod emit;

pub use emit::emit;
