//! optimizer — 架构无关优化器。
//!
//! 优化哲学（与用户深度讨论后定）：
//! - **不用模式匹配**（如硬编码 MatMul+Add→Linear）
//! - 用简单代数规则（x+0, x*1, x-x）+ 浮点结构优化（IEEE754 位级 trick、Flash Attention 式重排）
//! - IO 同样性（CSE 公共子表达式消除）
//! - 启发式 + cost model 决策融合
//!
//! 三阶段 pipeline（拆细→重排→融合）：
//! 1. **拆细**（[`decompose`]）：复杂算子（LayerNorm/Softmax/Gelu）→ 细粒度原语
//! 2. **重排**（[`algebra`] + [`float_opts`] + [`cse`]）：代数简化 + 浮点优化 + CSE
//! 3. **融合**（[`fuse`]）：基于 cost model 的多对一启发式融合

pub mod algebra;
pub mod constprop;
pub mod cost_model;
pub mod cse;
pub mod decompose;
pub mod float_opts;
pub mod fuse;
pub mod passes;
pub mod shape_infer;

use base::{Graph, Pass, PassContext, Result};

/// 架构无关 Pass 管理器
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add(&mut self, p: Box<dyn Pass>) -> &mut Self {
        self.passes.push(p);
        self
    }

    /// 构建三阶段 pipeline
    pub fn default_for(level: common::OptLevel, target: common::Target) -> Self {
        let mut pm = Self::new();
        pm.add(Box::new(passes::Verify));
        pm.add(Box::new(DecomposePass));
        pm.add(Box::new(ShapeInferPass));
        pm.add(Box::new(ConstPropPass));
        pm.add(Box::new(AlgebraPass));
        pm.add(Box::new(ConstPropPass));
        pm.add(Box::new(FloatOptPass));
        pm.add(Box::new(CsePass));
        pm.add(Box::new(passes::DeadCodeElim));
        if level >= common::OptLevel::O2 {
            pm.add(Box::new(FusionPass {
                coeffs: cost_model::CostCoeffs::for_target(target),
            }));
        }
        pm.add(Box::new(passes::DeadCodeElim));
        pm.add(Box::new(passes::Verify));
        pm
    }

    pub fn run(&mut self, graph: &mut Graph) -> Result<PassContext> {
        let mut ctx = PassContext::default();
        for p in &mut self.passes {
            p.run(graph, &mut ctx)?;
        }
        Ok(ctx)
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

struct DecomposePass;

impl Pass for DecomposePass {
    fn name(&self) -> &str {
        "decompose"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        let results = decompose::run_decompose(graph)?;
        ctx.inc("decompose_count");
        let _ = results;
        Ok(())
    }
}

struct ShapeInferPass;

impl Pass for ShapeInferPass {
    fn name(&self) -> &str {
        "shape-infer"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        let count = shape_infer::apply_shape_infer(graph)?;
        if count > 0 {
            ctx.inc("shape_infer_filled");
            ctx.stats.insert("shape_infer_count".into(), count);
        }
        Ok(())
    }
}

struct ConstPropPass;

impl Pass for ConstPropPass {
    fn name(&self) -> &str {
        "constprop"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        let count = constprop::apply_constprop(graph)?;
        if count > 0 {
            ctx.inc("constprop_applied");
            ctx.stats.insert("constprop_count".into(), count);
        }
        Ok(())
    }
}

struct AlgebraPass;

impl Pass for AlgebraPass {
    fn name(&self) -> &str {
        "algebra-simplify"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        let count = algebra::run_algebraic_simplify(graph)?;
        if count > 0 {
            ctx.inc("algebra_simplified");
            ctx.stats.insert("algebra_count".into(), count);
        }
        Ok(())
    }
}

struct FloatOptPass;

impl Pass for FloatOptPass {
    fn name(&self) -> &str {
        "float-opts"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        let count = float_opts::apply_float_opts(graph)?;
        if count > 0 {
            ctx.inc("float_opts_applied");
        }
        Ok(())
    }
}

struct CsePass;

impl Pass for CsePass {
    fn name(&self) -> &str {
        "cse"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        // 不动点迭代：CSE 消除节点后使用者 inputs 被重写，可能暴露新的
        // 公共子表达式机会，循环到无变化。设上限防意外死循环。
        let mut total = 0usize;
        for _ in 0..16 {
            let n = cse::apply_cse(graph)?;
            if n == 0 {
                break;
            }
            total += n;
        }
        if total > 0 {
            ctx.inc("cse_eliminated");
        }
        Ok(())
    }
}

struct FusionPass {
    coeffs: cost_model::CostCoeffs,
}

impl Pass for FusionPass {
    fn name(&self) -> &str {
        "fusion"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        let count = fuse::apply_fusion(graph, self.coeffs)?;
        if count > 0 {
            ctx.inc("fusion_applied");
        }
        Ok(())
    }
}
