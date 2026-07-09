//! dsl — 强类型 Python 子集 DSL（待用户提供样例后实现）

use base::{Graph, NeutronError, Result};

/// 解析 DSL 文本为架构无关图
pub fn parse(_src: &str) -> Result<Graph> {
    Err(NeutronError::Frontend("DSL 前端尚未实现，等待语法样例".into()))
}
