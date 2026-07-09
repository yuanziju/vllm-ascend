//! pt — PyTorch .pt 加载（占位）

use base::{Graph, NeutronError, Result};

/// 从 .pt 文件加载为架构无关图
pub fn parse(_bytes: &[u8]) -> Result<Graph> {
    Err(NeutronError::Frontend(".pt 前端尚未实现".into()))
}
