use super::pass::{Pass, PassContext, PassResult};

/// 嵌套优化器：持有一组 Pass，迭代至收敛
/// 嵌套深度 ≤ 3
#[derive(Debug)]
pub struct PassGroup {
    name: String,
    passes: Vec<Box<dyn Pass>>,
    max_iterations: usize,
    depth: usize,
}

impl PassGroup {
    pub fn new(name: impl Into<String>, depth: usize) -> Self {
        assert!(
            depth <= 3,
            "PassGroup nesting depth must be ≤ 3, got {}",
            depth
        );
        PassGroup {
            name: name.into(),
            passes: Vec::new(),
            max_iterations: 4,
            depth,
        }
    }

    pub fn with_pass(mut self, pass: impl Pass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl Pass for PassGroup {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&mut self, ctx: &mut PassContext) -> PassResult {
        let mut total_changed = false;
        let mut all_diags = Vec::new();

        for _iter in 0..self.max_iterations {
            let mut iter_changed = false;
            for pass in &mut self.passes {
                let result = pass.run(ctx);
                if result.changed {
                    iter_changed = true;
                    ctx.analyses.invalidate();
                }
                all_diags.extend(result.diagnostics);
            }
            if !iter_changed {
                break; // 收敛
            }
            total_changed = true;
        }

        PassResult {
            changed: total_changed,
            diagnostics: all_diags,
        }
    }
}