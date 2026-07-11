//! interference — 干扰图构建
//!
//! 干扰图是图着色寄存器分配的核心数据结构：
//! - 节点 = VReg
//! - 边 = 两个 VReg 的活跃区间重叠（不能共用物理寄存器）
//!
//! 干扰图是无向图：如果 A 干扰 B，则 B 也干扰 A。

use crate::liveness::LivenessResult;
use crate::types::{MachineInstr, VReg};
use std::collections::{HashMap, HashSet};

/// 干扰图
#[derive(Debug, Clone)]
pub struct InterferenceGraph {
    /// 邻接表：VReg → 所有干扰的 VReg 集合
    adj: HashMap<VReg, HashSet<VReg>>,
    /// 所有节点（包括无邻居的孤立节点）
    nodes: HashSet<VReg>,
}

impl Default for InterferenceGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl InterferenceGraph {
    pub fn new() -> Self {
        Self {
            adj: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, v: VReg) {
        self.nodes.insert(v);
        self.adj.entry(v).or_default();
    }

    /// 添加边（无向）
    pub fn add_edge(&mut self, a: VReg, b: VReg) {
        if a == b {
            return; // 自环无意义
        }
        self.nodes.insert(a);
        self.nodes.insert(b);
        self.adj.entry(a).or_default().insert(b);
        self.adj.entry(b).or_default().insert(a);
    }

    /// 获取节点的邻居集合引用
    /// 节点必须已通过 add_node 或 add_edge 加入图中，否则返回空集。
    pub fn neighbor_set(&self, v: VReg) -> &HashSet<VReg> {
        // adj 在 add_node 时一定 entry().or_default() 插入了空集，
        // 所以存在的节点一定有 entry。用 expect 明确不变式。
        self.adj
            .get(&v)
            .expect("neighbor_set: VReg must be added via add_node/add_edge first")
    }

    /// 节点的度数（邻居数）
    pub fn degree(&self, v: VReg) -> usize {
        self.adj.get(&v).map(|s| s.len()).unwrap_or(0)
    }

    /// 所有节点
    pub fn nodes(&self) -> &HashSet<VReg> {
        &self.nodes
    }

    /// 检查两个节点是否相邻
    pub fn adjacent(&self, a: VReg, b: VReg) -> bool {
        self.adj.get(&a).map(|s| s.contains(&b)).unwrap_or(false)
    }

    /// 节点数
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 移除节点（同时移除所有关联边）
    pub fn remove_node(&mut self, v: VReg) {
        if let Some(neighbors) = self.adj.remove(&v) {
            for n in &neighbors {
                if let Some(adj) = self.adj.get_mut(n) {
                    adj.remove(&v);
                }
            }
        }
        self.nodes.remove(&v);
    }

    /// 合并两个节点（coalescing 用）
    /// 将 b 的所有邻居合并到 a，然后移除 b
    pub fn merge(&mut self, a: VReg, b: VReg) {
        if a == b {
            return;
        }
        // 收集 b 的邻居
        let b_neighbors: Vec<VReg> = self
            .adj
            .get(&b)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        // 将 b 的邻居添加到 a
        for n in &b_neighbors {
            self.add_edge(a, *n);
        }

        // 移除 b
        self.remove_node(b);
    }
}

/// 从活跃分析结果构建干扰图
///
/// 遍历所有 VReg 对，如果活跃区间重叠则添加干扰边。
/// **Move 例外**：move 指令的 src/dst 如果只在 move 那一条指令上重叠，
/// 不加干扰边（它们可以共用寄存器，coalescing 会合并）。
/// 时间复杂度 O(V²) — 对于编译器后端可接受（VReg 数通常 < 1000）。
pub fn build(liveness: &LivenessResult, instructions: &[MachineInstr]) -> InterferenceGraph {
    let mut graph = InterferenceGraph::new();

    // 收集所有 VReg
    let vregs: Vec<VReg> = liveness.intervals.keys().copied().collect();
    for &v in &vregs {
        graph.add_node(v);
    }

    // 收集 move 对（src, dst）— 用于 move 例外
    let mut move_pairs: HashSet<(VReg, VReg)> = HashSet::new();
    for instr in instructions {
        if instr.is_move() {
            if let (Some(src), Some(dst)) = (instr.move_src(), instr.move_dst()) {
                move_pairs.insert((src, dst));
                move_pairs.insert((dst, src));
            }
        }
    }

    // 构建干扰边：所有活跃区间重叠的 VReg 对
    for i in 0..vregs.len() {
        for j in (i + 1)..vregs.len() {
            let a = vregs[i];
            let b = vregs[j];
            let interval_a = &liveness.intervals[&a];
            let interval_b = &liveness.intervals[&b];
            if !interval_a.overlaps(interval_b) {
                continue;
            }
            // Move 例外：如果 (a,b) 是 move 对且重叠区间只有单条指令，
            // 那条指令就是 move 本身，不加干扰边（允许 coalescing 合并）
            if move_pairs.contains(&(a, b)) {
                let overlap_start = interval_a.start.max(interval_b.start);
                let overlap_end = interval_a.end.min(interval_b.end);
                if overlap_start == overlap_end {
                    // 单点重叠 = move 指令 → 跳过
                    continue;
                }
            }
            graph.add_edge(a, b);
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::analyze;
    use crate::types::*;

    fn mk_instr(op: &str, uses: Vec<u32>, defs: Vec<u32>) -> MachineInstr {
        MachineInstr {
            op: op.to_string(),
            operands: uses.into_iter().map(VReg).map(Operand::VReg).collect(),
            defs: defs.into_iter().map(VReg).map(Operand::VReg).collect(),
            args: vec![],
        }
    }

    #[test]
    fn build_graph_basic() {
        // v0 [0,2], v1 [1,3] → 干扰
        // v2 [4,5] → 不与任何人干扰
        let instrs = vec![
            mk_instr("load", vec![], vec![0]),    // v0 def @ 0
            mk_instr("load", vec![], vec![1]),    // v1 def @ 1
            mk_instr("fadd", vec![0, 1], vec![]), // v0,v1 use @ 2
            mk_instr("load", vec![], vec![2]),    // v2 def @ 3
            mk_instr("store", vec![2], vec![]),   // v2 use @ 4
        ];

        let liveness = analyze(&instrs);
        let graph = build(&liveness, &instrs);

        // v0 和 v1 干扰
        assert!(graph.adjacent(VReg(0), VReg(1)));
        // v0 和 v2 不干扰
        assert!(!graph.adjacent(VReg(0), VReg(2)));
        // v1 和 v2 不干扰
        assert!(!graph.adjacent(VReg(1), VReg(2)));
    }

    #[test]
    fn degree_computation() {
        // v0 干扰 v1, v2, v3 → degree = 3
        let mut g = InterferenceGraph::new();
        g.add_edge(VReg(0), VReg(1));
        g.add_edge(VReg(0), VReg(2));
        g.add_edge(VReg(0), VReg(3));

        assert_eq!(g.degree(VReg(0)), 3);
        assert_eq!(g.degree(VReg(1)), 1);
        assert_eq!(g.degree(VReg(4)), 0); // 不存在
    }

    #[test]
    fn remove_node_removes_edges() {
        let mut g = InterferenceGraph::new();
        g.add_edge(VReg(0), VReg(1));
        g.add_edge(VReg(0), VReg(2));
        g.add_edge(VReg(1), VReg(2));

        g.remove_node(VReg(0));

        assert!(!g.nodes().contains(&VReg(0)));
        assert!(!g.adjacent(VReg(1), VReg(0)));
        assert!(!g.adjacent(VReg(2), VReg(0)));
        // v1-v2 边应该还在
        assert!(g.adjacent(VReg(1), VReg(2)));
    }

    #[test]
    fn merge_nodes() {
        let mut g = InterferenceGraph::new();
        // v0 → v1, v2
        // v3 → v2
        g.add_edge(VReg(0), VReg(1));
        g.add_edge(VReg(0), VReg(2));
        g.add_edge(VReg(3), VReg(2));

        // 合并 v0 和 v3：v3 的邻居(v2) 加到 v0
        g.merge(VReg(0), VReg(3));

        // v0 现在应该干扰 v1, v2
        assert!(g.adjacent(VReg(0), VReg(1)));
        assert!(g.adjacent(VReg(0), VReg(2)));
        // v3 被移除
        assert!(!g.nodes().contains(&VReg(3)));
    }

    #[test]
    fn no_self_loops() {
        let mut g = InterferenceGraph::new();
        g.add_edge(VReg(0), VReg(0));
        assert_eq!(g.degree(VReg(0)), 0);
    }
}
