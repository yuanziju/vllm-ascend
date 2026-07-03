use crate::pass::group::PassGroup;
use crate::pass::manager::PassManager;
use crate::pass::transforms::canonicalize::CanonicalizePass;
use crate::pass::transforms::cse::CSEPass;
use crate::pass::transforms::dce::DCEPass;

/// 优化级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
}

/// 优化策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptStrategy {
    Conservative,
    Aggressive,
}

/// 预定义的 Pass Pipeline
pub struct PassPipeline;

impl PassPipeline {
    /// 根据级别和策略构建 PassManager
    pub fn build(level: OptLevel, strategy: OptStrategy) -> PassManager {
        let mut manager = PassManager::new();

        match level {
            OptLevel::O0 => {
                // 仅必要转换
                // manager.add_pass(ShapeInfer::new());
            }
            OptLevel::O1 => {
                manager.add_pass(CanonicalizePass::new());
                manager.add_pass(CSEPass::new());
                manager.add_pass(DCEPass::new());
                if strategy == OptStrategy::Aggressive {
                    // manager.add_pass(Inliner::new());
                }
                // manager.add_pass(ShapeInfer::new());
            }
            OptLevel::O2 => {
                manager.add_pass(CanonicalizePass::new());
                manager.add_pass(CSEPass::new());

                // 融合组（嵌套优化器，迭代至收敛）
                let mut fusion_group = PassGroup::new("fusion-group", 1);
                // fusion_group = fusion_group.with_pass(Fusion::new());
                // fusion_group = fusion_group.with_pass(LayoutOpt::new());
                fusion_group = fusion_group.with_max_iterations(3);
                manager.add_pass(fusion_group);

                manager.add_pass(DCEPass::new());

                if strategy == OptStrategy::Aggressive {
                    // manager.add_pass(Inliner::new());
                    // manager.add_pass(Tiling::new());
                }
                // manager.add_pass(ShapeInfer::new());
            }
            OptLevel::O3 => {
                // O2 + 激进优化
                let mgr = Self::build(OptLevel::O2, strategy);
                // mgr.add_pass(LoopUnroll::new());
                // mgr.add_pass(Vectorize::new());
                if strategy == OptStrategy::Aggressive {
                    // mgr.add_pass(Quantize::new());
                }
                return mgr;
            }
        }

        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_o1_conservative_pipeline() {
        let mut manager = PassPipeline::build(OptLevel::O1, OptStrategy::Conservative);
        // 验证 pipeline 构建成功（不 panic）
        let mut graph = crate::graph::ir::IrGraph::new();
        let result = manager.run(&mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_o2_aggressive_pipeline() {
        let mut manager = PassPipeline::build(OptLevel::O2, OptStrategy::Aggressive);
        let mut graph = crate::graph::ir::IrGraph::new();
        let result = manager.run(&mut graph);
        assert!(result.is_ok());
    }
}