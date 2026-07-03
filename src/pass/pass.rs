use crate::graph::ir::IrGraph;
use super::diagnostics::Diagnostic;

/// Pass 执行结果
#[derive(Debug, Clone)]
pub struct PassResult {
    pub changed: bool,
    pub diagnostics: Vec<Diagnostic>,
}

impl PassResult {
    pub fn unchanged() -> Self {
        PassResult {
            changed: false,
            diagnostics: Vec::new(),
        }
    }
    pub fn changed() -> Self {
        PassResult {
            changed: true,
            diagnostics: Vec::new(),
        }
    }
    pub fn with_diag(mut self, d: Diagnostic) -> Self {
        self.diagnostics.push(d);
        self
    }
}

/// 惰性分析缓存
#[derive(Debug, Default)]
pub struct AnalysisCache {
    // 后续扩展：Dominance, Liveness, UseDef, ShapeInfer 结果
    invalidated: bool,
}

impl AnalysisCache {
    pub fn new() -> Self {
        AnalysisCache { invalidated: false }
    }
    pub fn invalidate(&mut self) {
        self.invalidated = true;
    }
    pub fn is_valid(&self) -> bool {
        !self.invalidated
    }
    pub fn validate(&mut self) {
        self.invalidated = false;
    }
}

/// Pass 上下文
pub struct PassContext<'a> {
    pub graph: &'a mut IrGraph,
    pub analyses: &'a mut AnalysisCache,
    pub diagnostics: Vec<Diagnostic>,
}

impl<'a> PassContext<'a> {
    pub fn new(graph: &'a mut IrGraph, analyses: &'a mut AnalysisCache) -> Self {
        PassContext {
            graph,
            analyses,
            diagnostics: Vec::new(),
        }
    }
    pub fn emit(&mut self, d: Diagnostic) {
        self.diagnostics.push(d);
    }
}

/// Pass trait
pub trait Pass: std::fmt::Debug {
    fn name(&self) -> &str;
    fn run(&mut self, ctx: &mut PassContext) -> PassResult;
}

/// 动态 Pass 包装
pub type PassBox = Box<dyn Pass>;