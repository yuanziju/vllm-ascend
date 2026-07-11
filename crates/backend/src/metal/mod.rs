//! metal — Metal Shading Language (MSL) 后端代码生成
//!
//! 生成 `.metal` kernel 源码，支持：
//! - simdgroup_matrix（8x8 矩阵乘法单元）
//! - threadgroup memory（shared memory 等价物）
//! - Apple GPU family（M1/M2/M3）

pub mod emit;

pub use emit::emit;
