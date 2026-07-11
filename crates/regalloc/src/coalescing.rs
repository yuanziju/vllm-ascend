//! coalescing — 保守合并（Conservative Briggs Coalescing）
//!
//! 合并 move-related VReg 对（move 指令的源和目标），减少寄存器间 move 指令。
//! 保守策略：仅当合并后节点的度数 < K（可用寄存器数）时才合并，
//! 确保合并不引入新的溢出。
//!
//! 参考：Briggs (1994) — "Register Allocation via Graph Coloring"

use crate::interference::InterferenceGraph;
use crate::types::*;
use std::collections::HashSet;

/// 合并候选：move 指令的 (src, dst) VReg 对
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoalescePair {
    pub src: VReg,
    pub dst: VReg,
}

/// 保守 Briggs 合并
///
/// 参数：
/// - graph: 干扰图（会被修改）
/// - pairs: 合并候选列表
/// - k: 可用物理寄存器数
/// - threshold: 合并阈值（合并后度数 < threshold 才合并）
///
/// 返回成功合并的对列表。未合并的对保留在图中不处理。
pub fn coalesce(
    graph: &mut InterferenceGraph,
    pairs: &[CoalescePair],
    _k: usize,
    threshold: usize,
) -> Vec<CoalescePair> {
    let mut merged = Vec::new();
    let mut merged_set: HashSet<VReg> = HashSet::new();

    for &pair in pairs {
        let CoalescePair { src, dst } = pair;

        // 已被合并的节点跳过
        if merged_set.contains(&src) || merged_set.contains(&dst) {
            continue;
        }

        // 如果 src 和 dst 已经干扰（不能合并同一个寄存器）
        if graph.adjacent(src, dst) {
            continue;
        }

        // 保守判定：合并后的度数 = neighbors(src) ∪ neighbors(dst) 的大小
        let src_neighbors: HashSet<VReg> = graph.neighbor_set(src).iter().copied().collect();
        let dst_neighbors: HashSet<VReg> = graph.neighbor_set(dst).iter().copied().collect();

        let mut combined: HashSet<VReg> = src_neighbors.clone();
        combined.extend(dst_neighbors.iter());
        // 合并后的度数 = combined 中既不是 src 也不是 dst 的节点数
        let combined_degree = combined.iter().filter(|&&v| v != src && v != dst).count();

        // 保守 Briggs：合并后度数 < threshold 才安全合并
        if combined_degree < threshold {
            graph.merge(src, dst);
            merged.push(pair);
            merged_set.insert(dst);
        }
    }

    merged
}

/// 从 MachineInstr 序列中提取合并候选（move 指令的 src→dst 对）
pub fn find_coalesce_pairs(instructions: &[MachineInstr]) -> Vec<CoalescePair> {
    let mut pairs = Vec::new();
    for instr in instructions {
        if instr.is_move() {
            if let (Some(src), Some(dst)) = (instr.move_src(), instr.move_dst()) {
                pairs.push(CoalescePair { src, dst });
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interference::InterferenceGraph;

    #[test]
    fn coalesce_non_interfering() {
        // v0 和 v1 不干扰，度数都为 0 → 合并安全
        let mut g = InterferenceGraph::new();
        g.add_node(VReg(0));
        g.add_node(VReg(1));

        let pairs = vec![CoalescePair {
            src: VReg(0),
            dst: VReg(1),
        }];
        let merged = coalesce(&mut g, &pairs, 4, 3);

        assert_eq!(merged.len(), 1);
        assert!(!g.nodes().contains(&VReg(1))); // dst 被移除
        assert!(g.nodes().contains(&VReg(0))); // src 保留
    }

    #[test]
    fn coalesce_skips_interfering() {
        // v0 和 v1 互相干扰 → 不合并
        let mut g = InterferenceGraph::new();
        g.add_edge(VReg(0), VReg(1));

        let pairs = vec![CoalescePair {
            src: VReg(0),
            dst: VReg(1),
        }];
        let merged = coalesce(&mut g, &pairs, 4, 3);

        assert_eq!(merged.len(), 0);
        assert!(g.nodes().contains(&VReg(0)));
        assert!(g.nodes().contains(&VReg(1)));
    }

    #[test]
    fn coalesce_conservative_blocks_high_degree() {
        // v0 有 3 个邻居(v1,v2,v3)，v3 有 0 个邻居
        // 合并 v0+v3 后度数 = 3（v1,v2,v3 被合并移除，但 v1,v2,v3 邻居保留）
        // threshold=3 → 3 < 3 false → 不合并
        let mut g = InterferenceGraph::new();
        g.add_edge(VReg(0), VReg(1));
        g.add_edge(VReg(0), VReg(2));
        g.add_edge(VReg(0), VReg(3));
        g.add_node(VReg(4));

        let pairs = vec![CoalescePair {
            src: VReg(0),
            dst: VReg(4),
        }];
        let merged = coalesce(&mut g, &pairs, 4, 3);

        // 合并后度数 = 3（v0 的 3 个邻居），3 < 3 false → 不合并
        assert_eq!(merged.len(), 0);
    }

    #[test]
    fn coalesce_allows_low_degree() {
        // v0 有 1 个邻居(v1)，v4 有 0 个邻居
        // 合并 v0+v4 后度数 = 1（v1），1 < 3 → 合并
        let mut g = InterferenceGraph::new();
        g.add_edge(VReg(0), VReg(1));
        g.add_node(VReg(4));

        let pairs = vec![CoalescePair {
            src: VReg(0),
            dst: VReg(4),
        }];
        let merged = coalesce(&mut g, &pairs, 4, 3);

        assert_eq!(merged.len(), 1);
        assert!(g.nodes().contains(&VReg(0)));
        assert!(!g.nodes().contains(&VReg(4)));
    }

    #[test]
    fn find_coalesce_pairs_from_moves() {
        let instrs = vec![
            MachineInstr {
                op: "load".into(),
                operands: vec![],
                defs: vec![Operand::VReg(VReg(0))],
                args: vec![],
            },
            MachineInstr {
                op: "mov".into(),
                operands: vec![Operand::VReg(VReg(0))],
                defs: vec![Operand::VReg(VReg(1))],
                args: vec![],
            },
            MachineInstr {
                op: "mov".into(),
                operands: vec![Operand::VReg(VReg(1))],
                defs: vec![Operand::VReg(VReg(2))],
                args: vec![],
            },
        ];

        let pairs = find_coalesce_pairs(&instrs);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].src, VReg(0));
        assert_eq!(pairs[0].dst, VReg(1));
        assert_eq!(pairs[1].src, VReg(1));
        assert_eq!(pairs[1].dst, VReg(2));
    }
}
