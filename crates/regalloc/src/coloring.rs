//! coloring — 图着色寄存器分配（Chaitin-Briggs 算法）
//!
//! 算法流程：
//! 1. Simplify：反复移除度数 < K 的节点入栈（这些节点一定能着色）
//! 2. Spill：如果所有节点度数 >= K，选择溢出代价最低的节点标记为溢出
//! 3. Select：从栈中弹出节点，贪心分配最小的可用颜色（物理寄存器）
//!
//! 溢出代价模型：cost = (uses + defs) / interval_size
//! — 使用次数少、活跃区间短的 VReg 优先溢出（代价低）

use crate::interference::InterferenceGraph;
use crate::liveness::LivenessResult;
use crate::types::*;
use std::collections::{HashMap, HashSet};

/// 着色结果
#[derive(Debug, Clone)]
pub struct ColoringResult {
    /// VReg → PReg 映射
    pub assignment: HashMap<VReg, PReg>,
    /// 被溢出的 VReg 集合
    pub spilled: HashSet<VReg>,
}

/// 可分配的物理寄存器列表（按 PReg 索引排序）
type Colors = Vec<PReg>;

/// Chaitin-Briggs 图着色
///
/// 参数：
/// - graph: 干扰图
/// - liveness: 活跃分析结果（用于溢出代价计算）
/// - colors: 可分配的物理寄存器列表
///
/// 返回着色结果（每个 VReg 分配到一个 PReg，或被标记为溢出）
pub fn color(
    graph: &InterferenceGraph,
    liveness: &LivenessResult,
    colors: &Colors,
) -> ColoringResult {
    let k = colors.len();
    let mut work_graph = graph.clone();
    let mut stack: Vec<VReg> = Vec::new();
    let mut spilled: HashSet<VReg> = HashSet::new();

    // Phase 1: Simplify + Spill
    loop {
        // Simplify：找度数 < K 的节点
        let low_degree = work_graph
            .nodes()
            .iter()
            .copied()
            .find(|&v| work_graph.degree(v) < k);

        if let Some(node) = low_degree {
            // 移除低度节点入栈
            work_graph.remove_node(node);
            stack.push(node);
        } else if work_graph.is_empty() {
            break;
        } else {
            // 所有节点度数 >= K → 溢出代价最低的
            let spill_candidate = work_graph.nodes().iter().copied().min_by(|a, b| {
                let cost_a = spill_cost(*a, liveness);
                let cost_b = spill_cost(*b, liveness);
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(candidate) = spill_candidate {
                spilled.insert(candidate);
                work_graph.remove_node(candidate);
                stack.push(candidate);
            } else {
                break;
            }
        }
    }

    // Phase 2: Select（贪心着色）
    let mut assignment: HashMap<VReg, PReg> = HashMap::new();

    for &vreg in stack.iter().rev() {
        if spilled.contains(&vreg) {
            continue; // 溢出的 VReg 不着色
        }

        // 收集已着色邻居占用的颜色
        let mut used_colors: HashSet<u32> = HashSet::new();
        // 需要从原图查邻居（work_graph 已被 simplify 破坏）
        for &neighbor in graph.neighbor_set(vreg) {
            if let Some(&preg) = assignment.get(&neighbor) {
                used_colors.insert(preg.0);
            }
        }

        // 分配最小的可用颜色
        if let Some(&color) = colors.iter().find(|c| !used_colors.contains(&c.0)) {
            assignment.insert(vreg, color);
        } else {
            // 没有可用颜色 → 也溢出
            spilled.insert(vreg);
        }
    }

    ColoringResult {
        assignment,
        spilled,
    }
}

/// 计算溢出代价
///
/// cost = (uses + defs) / interval_size
/// — 使用次数多、活跃区间短的 VReg 溢出代价高（不应该溢出）
/// — 使用次数少、活跃区间长的 VReg 溢出代价低（适合溢出）
fn spill_cost(vreg: VReg, liveness: &LivenessResult) -> f64 {
    let use_count = liveness.use_count(vreg) as f64;
    let interval = liveness.get(vreg).map(|iv| iv.len() as f64).unwrap_or(1.0);
    if interval == 0.0 {
        use_count + 1.0 // 避免 cost = 0 被优先溢出
    } else {
        (use_count + 1.0) / interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interference::build;
    use crate::liveness::analyze;

    fn mk_instr(op: &str, uses: Vec<u32>, defs: Vec<u32>) -> MachineInstr {
        MachineInstr {
            op: op.to_string(),
            operands: uses.into_iter().map(VReg).map(Operand::VReg).collect(),
            defs: defs.into_iter().map(VReg).map(Operand::VReg).collect(),
            args: vec![],
        }
    }

    fn colors(n: u32) -> Colors {
        (0..n).map(PReg).collect()
    }

    #[test]
    fn color_simple_no_spill() {
        // 3 个 VReg，两两干扰（三角形），K=3 → 不溢出
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("load", vec![], vec![1]),
            mk_instr("load", vec![], vec![2]),
            mk_instr("fadd", vec![0, 1, 2], vec![]),
        ];

        let liveness = analyze(&instrs);
        let graph = build(&liveness, &instrs);
        let cols = colors(3);

        let result = color(&graph, &liveness, &cols);

        assert_eq!(result.assignment.len(), 3);
        assert!(result.spilled.is_empty());

        // 三个互相干扰的 VReg 必须分配不同颜色
        let p0 = result.assignment[&VReg(0)];
        let p1 = result.assignment[&VReg(1)];
        let p2 = result.assignment[&VReg(2)];
        assert_ne!(p0, p1);
        assert_ne!(p0, p2);
        assert_ne!(p1, p2);
    }

    #[test]
    fn color_reuses_register_when_no_interference() {
        // v0 [0,0] 和 v1 [2,2] 不干扰 → 可共用同一寄存器
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("load", vec![], vec![1]), // v1 def @ 1, 与 v0 不干扰
            mk_instr("store", vec![1], vec![]),
        ];

        let liveness = analyze(&instrs);
        let graph = build(&liveness, &instrs);
        let cols = colors(1); // 只有 1 个寄存器

        let result = color(&graph, &liveness, &cols);

        assert!(result.spilled.is_empty());
        // v0 和 v1 应该共用 R0
        assert_eq!(result.assignment[&VReg(0)], PReg(0));
        assert_eq!(result.assignment[&VReg(1)], PReg(0));
    }

    #[test]
    fn color_spills_when_not_enough_registers() {
        // 4 个互相干扰的 VReg，K=2 → 必须溢出至少 2 个
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),
            mk_instr("load", vec![], vec![1]),
            mk_instr("load", vec![], vec![2]),
            mk_instr("load", vec![], vec![3]),
            mk_instr("fadd", vec![0, 1, 2, 3], vec![]),
        ];

        let liveness = analyze(&instrs);
        let graph = build(&liveness, &instrs);
        let cols = colors(2);

        let result = color(&graph, &liveness, &cols);

        // 至少溢出 2 个
        assert!(result.spilled.len() >= 2);
        // 着色的不超过 2 个（K=2）
        assert!(result.assignment.len() <= 2);
    }

    #[test]
    fn spill_cost_favors_low_use() {
        // v0 使用 3 次（密集），v1 使用 1 次（稀疏）→ v1 密度低 → 优先溢出
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),   // v0 def @ 0
            mk_instr("load", vec![], vec![1]),   // v1 def @ 1
            mk_instr("use", vec![0], vec![]),    // v0 use @ 2
            mk_instr("use", vec![0], vec![]),    // v0 use @ 3
            mk_instr("use", vec![0, 1], vec![]), // v0,v1 use @ 4
        ];

        let liveness = analyze(&instrs);

        // v0 使用 3 次，v1 使用 1 次
        assert_eq!(liveness.use_count(VReg(0)), 3);
        assert_eq!(liveness.use_count(VReg(1)), 1);

        // v1 的溢出代价应该比 v0 低（密度 = (uses+1)/interval_len 更低 → 优先溢出）
        let cost_v0 = spill_cost(VReg(0), &liveness);
        let cost_v1 = spill_cost(VReg(1), &liveness);
        assert!(
            cost_v1 < cost_v0,
            "v1 cost {} should be < v0 cost {}",
            cost_v1,
            cost_v0
        );
    }

    #[test]
    fn color_empty_graph() {
        let liveness = LivenessResult::default();
        let graph = InterferenceGraph::new();
        let cols = colors(4);

        let result = color(&graph, &liveness, &cols);
        assert!(result.assignment.is_empty());
        assert!(result.spilled.is_empty());
    }
}
