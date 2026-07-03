use crate::graph::ir::IrGraph;
use crate::pass::manager::PassManager;
use crate::pass::pipeline::{OptLevel, OptStrategy, PassPipeline};

/// 编译器：JVM 风格，构造即配置，运行时不可变
pub struct Compiler {
    pipeline: PassManager,
}

impl Compiler {
    /// 创建编译器 Builder
    pub fn builder() -> CompilerBuilder {
        CompilerBuilder::new()
    }

    /// 编译 IR（架构无关优化）
    pub fn compile(&mut self, graph: &mut IrGraph) -> Result<(), String> {
        self.pipeline.run(graph).map_err(|results| {
            format!(
                "compilation failed with {} pass results",
                results.len()
            )
        })
    }
}

/// Compiler 构造器
pub struct CompilerBuilder {
    level: OptLevel,
    strategy: OptStrategy,
}

impl CompilerBuilder {
    pub fn new() -> Self {
        CompilerBuilder {
            level: OptLevel::O1,
            strategy: OptStrategy::Conservative,
        }
    }

    pub fn opt_level(mut self, level: OptLevel) -> Self {
        self.level = level;
        self
    }

    pub fn strategy(mut self, strategy: OptStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn build(self) -> Compiler {
        Compiler {
            pipeline: PassPipeline::build(self.level, self.strategy),
        }
    }
}

impl Default for CompilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_builder() {
        let mut compiler = Compiler::builder()
            .opt_level(OptLevel::O2)
            .strategy(OptStrategy::Aggressive)
            .build();

        let mut graph = IrGraph::new();
        assert!(compiler.compile(&mut graph).is_ok());
    }

    #[test]
    fn test_compiler_default() {
        let mut compiler = Compiler::builder().build();
        let mut graph = IrGraph::new();
        assert!(compiler.compile(&mut graph).is_ok());
    }
}