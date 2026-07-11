//! cann — Ascend CANN C++ 后端代码生成
//!
//! 生成 Ascend NPU 算子 C++ 源码，支持：
//! - Vector Core：SIMD 向量指令（Add/Mul/Sqrt 等）
//! - Cube Core：矩阵乘法单元（MatMul）
//! - Ascend C++ API：`__aicore__` kernel + DataCopy + Broadcast

pub mod emit;

pub use emit::emit;
