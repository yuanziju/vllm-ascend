//! frontend — 前端：ONNX / DSL / .pt → 架构无关图

pub mod dsl;
pub mod onnx;
pub mod pt;

pub use onnx::parse as parse_onnx;
