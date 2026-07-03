use super::pass::{Pass, PassContext, PassResult, AnalysisCache};
use crate::graph::ir::IrGraph;

/// Pass 管理器：按序执行 Pass，管理分析缓存
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    analysis_cache: AnalysisCache,
}

impl PassManager {
    pub fn new() -> Self {
        PassManager {
            passes: Vec::new(),
            analysis_cache: AnalysisCache::new(),
        }
    }

    pub fn add_pass(&mut self, pass: impl Pass + 'static) {
        self.passes.push(Box::new(pass));
    }

    pub fn run(&mut self, graph: &mut IrGraph) -> Result<(), Vec<PassResult>> {
        let mut results = Vec::new();
        self.analysis_cache.invalidate();

        for pass in &mut self.passes {
            let mut ctx = PassContext::new(graph, &mut self.analysis_cache);
            let result = pass.run(&mut ctx);
            results.push(result);
        }

        Ok(())
    }
}