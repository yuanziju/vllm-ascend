//! onnx — ONNX 解析（MVP 占位，后续接入 prost）

use base::{Graph, NeutronError, OpKind, Result};

/// 解析 ONNX 字节流为架构无关图
pub fn parse(_bytes: &[u8]) -> Result<Graph> {
    // TODO: 接入 prost + ONNX schema
    let mut g = Graph::new("onnx");
    let _n = g.add_node(OpKind::Placeholder);
    if _bytes.is_empty() {
        return Ok(g);
    }
    Err(NeutronError::Frontend("ONNX 解析尚未实现".into()))
}
