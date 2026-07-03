//! 有向图底层存储 —— 连续内存 + 代际索引 + unsafe 零开销抽象。
//!
//! 设计目标：
//! - 节点和边全部存储在连续 `Vec` 中，缓存友好。
//! - 使用代际索引 (generational index) 防止 use-after-free。
//! - 邻接链表通过索引（而非指针）串联，所有边数据在同一块连续内存中。
//! - 双向链表边实现 O(1) 删除。
//! - 空闲槽位复用，避免内存碎片。

use std::fmt;

const NULL_EDGE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeIndex {
    pub index: u32,
    pub generation: u32,
}

impl NodeIndex {
    #[inline]
    pub const fn index(self) -> u32 { self.index }
    #[inline]
    pub const fn generation(self) -> u32 { self.generation }
}

impl EdgeIndex {
    #[inline]
    pub const fn index(self) -> u32 { self.index }
    #[inline]
    pub const fn generation(self) -> u32 { self.generation }
}

impl fmt::Debug for NodeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({}, gen={})", self.index, self.generation)
    }
}

impl fmt::Debug for EdgeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge({}, gen={})", self.index, self.generation)
    }
}

struct NodeEntry<N> {
    data: N,
    first_outgoing: u32,
    first_incoming: u32,
}

struct EdgeEntry<E> {
    data: E,
    source: u32,
    target: u32,
    prev_outgoing: u32,
    next_outgoing: u32,
    prev_incoming: u32,
    next_incoming: u32,
}

pub struct Graph<N, E> {
    nodes: Vec<NodeEntry<N>>,
    edges: Vec<EdgeEntry<E>>,
    free_nodes: Vec<u32>,
    free_edges: Vec<u32>,
    node_generations: Vec<u32>,
    edge_generations: Vec<u32>,
}

impl<N, E> Graph<N, E> {
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            free_nodes: Vec::new(),
            free_edges: Vec::new(),
            node_generations: Vec::new(),
            edge_generations: Vec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(nodes),
            edges: Vec::with_capacity(edges),
            free_nodes: Vec::new(),
            free_edges: Vec::new(),
            node_generations: Vec::with_capacity(nodes),
            edge_generations: Vec::with_capacity(edges),
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free_nodes.len()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len() - self.free_edges.len()
    }

    #[inline]
    pub fn reserve(&mut self, nodes: usize, edges: usize) {
        self.nodes.reserve(nodes);
        self.edges.reserve(edges);
        self.node_generations.reserve(nodes);
        self.edge_generations.reserve(edges);
    }

    pub fn clear(&mut self) {
        for edge in self.edges.iter_mut() {
            unsafe { std::ptr::drop_in_place(&mut edge.data); }
        }
        for node in self.nodes.iter_mut() {
            unsafe { std::ptr::drop_in_place(&mut node.data); }
        }
        unsafe {
            self.nodes.set_len(0);
            self.edges.set_len(0);
            self.node_generations.set_len(0);
            self.edge_generations.set_len(0);
        }
        self.free_nodes.clear();
        self.free_edges.clear();
    }

    pub fn add_node(&mut self, data: N) -> NodeIndex {
        if let Some(free_idx) = self.free_nodes.pop() {
            let idx = free_idx as usize;
            self.node_generations[idx] = self.node_generations[idx].wrapping_add(1);
            let generation = self.node_generations[idx];
            unsafe {
                let slot = self.nodes.as_mut_ptr().add(idx);
                std::ptr::write(slot, NodeEntry {
                    data,
                    first_outgoing: NULL_EDGE,
                    first_incoming: NULL_EDGE,
                });
            }
            NodeIndex { index: free_idx, generation }
        } else {
            let idx = self.nodes.len() as u32;
            self.node_generations.push(1);
            self.nodes.push(NodeEntry {
                data,
                first_outgoing: NULL_EDGE,
                first_incoming: NULL_EDGE,
            });
            NodeIndex { index: idx, generation: 1 }
        }
    }

    pub fn remove_node(&mut self, node: NodeIndex) -> Option<N> {
        if !self.contains_node(node) {
            return None;
        }
        let idx = node.index as usize;
        while self.nodes[idx].first_outgoing != NULL_EDGE {
            let e = self.nodes[idx].first_outgoing;
            self.remove_edge_raw(e);
        }
        while self.nodes[idx].first_incoming != NULL_EDGE {
            let e = self.nodes[idx].first_incoming;
            self.remove_edge_raw(e);
        }
        let data = unsafe { std::ptr::read(&self.nodes[idx].data) };
        self.node_generations[idx] = self.node_generations[idx].wrapping_add(1);
        self.free_nodes.push(node.index);
        Some(data)
    }

    #[inline]
    pub fn contains_node(&self, node: NodeIndex) -> bool {
        let idx = node.index as usize;
        idx < self.nodes.len()
            && self.node_generations[idx] == node.generation
            && self.node_generations[idx] != 0
    }

    #[inline]
    pub fn node(&self, node: NodeIndex) -> Option<&N> {
        if !self.contains_node(node) { return None; }
        Some(&self.nodes[node.index as usize].data)
    }

    #[inline]
    pub fn node_mut(&mut self, node: NodeIndex) -> Option<&mut N> {
        if !self.contains_node(node) { return None; }
        Some(&mut self.nodes[node.index as usize].data)
    }

    #[inline]
    pub unsafe fn node_unchecked(&self, node: NodeIndex) -> &N {
        unsafe { &self.nodes.get_unchecked(node.index as usize).data }
    }

    #[inline]
    pub unsafe fn node_mut_unchecked(&mut self, node: NodeIndex) -> &mut N {
        unsafe { &mut self.nodes.get_unchecked_mut(node.index as usize).data }
    }

    pub fn degree_out(&self, node: NodeIndex) -> Option<usize> {
        if !self.contains_node(node) { return None; }
        Some(self.outgoing_edges_raw(node).count())
    }

    pub fn degree_in(&self, node: NodeIndex) -> Option<usize> {
        if !self.contains_node(node) { return None; }
        Some(self.incoming_edges_raw(node).count())
    }

    pub fn add_edge(&mut self, source: NodeIndex, target: NodeIndex, data: E) -> Option<EdgeIndex> {
        if !self.contains_node(source) || !self.contains_node(target) {
            return None;
        }
        let s_idx = source.index as usize;
        let t_idx = target.index as usize;
        let edge_idx = if let Some(free_idx) = self.free_edges.pop() {
            let idx = free_idx as usize;
            self.edge_generations[idx] = self.edge_generations[idx].wrapping_add(1);
            let generation = self.edge_generations[idx];
            unsafe {
                let slot = self.edges.as_mut_ptr().add(idx);
                std::ptr::write(slot, EdgeEntry {
                    data,
                    source: source.index,
                    target: target.index,
                    prev_outgoing: NULL_EDGE,
                    next_outgoing: self.nodes[s_idx].first_outgoing,
                    prev_incoming: NULL_EDGE,
                    next_incoming: self.nodes[t_idx].first_incoming,
                });
            }
            EdgeIndex { index: free_idx, generation }
        } else {
            let idx = self.edges.len() as u32;
            self.edge_generations.push(1);
            self.edges.push(EdgeEntry {
                data,
                source: source.index,
                target: target.index,
                prev_outgoing: NULL_EDGE,
                next_outgoing: self.nodes[s_idx].first_outgoing,
                prev_incoming: NULL_EDGE,
                next_incoming: self.nodes[t_idx].first_incoming,
            });
            EdgeIndex { index: idx, generation: 1 }
        };
        let _e_idx = edge_idx.index as usize;
        if self.nodes[s_idx].first_outgoing != NULL_EDGE {
            let old_head = self.nodes[s_idx].first_outgoing as usize;
            self.edges[old_head].prev_outgoing = edge_idx.index;
        }
        self.nodes[s_idx].first_outgoing = edge_idx.index;
        if self.nodes[t_idx].first_incoming != NULL_EDGE {
            let old_head = self.nodes[t_idx].first_incoming as usize;
            self.edges[old_head].prev_incoming = edge_idx.index;
        }
        self.nodes[t_idx].first_incoming = edge_idx.index;
        Some(edge_idx)
    }

    pub fn remove_edge(&mut self, edge: EdgeIndex) -> Option<E> {
        if !self.contains_edge(edge) { return None; }
        Some(self.remove_edge_raw(edge.index))
    }

    fn remove_edge_raw(&mut self, edge: u32) -> E {
        let idx = edge as usize;
        let edges_ptr = self.edges.as_mut_ptr();
        let nodes_ptr = self.nodes.as_mut_ptr();
        let gens_ptr = self.edge_generations.as_mut_ptr();
        let (source, target, prev_out, next_out, prev_in, next_in) = unsafe {
            let e = &*edges_ptr.add(idx);
            (e.source, e.target, e.prev_outgoing, e.next_outgoing,
             e.prev_incoming, e.next_incoming)
        };
        unsafe {
            if prev_out == NULL_EDGE {
                (*nodes_ptr.add(source as usize)).first_outgoing = next_out;
            } else {
                (*edges_ptr.add(prev_out as usize)).next_outgoing = next_out;
            }
            if next_out != NULL_EDGE {
                (*edges_ptr.add(next_out as usize)).prev_outgoing = prev_out;
            }
            if prev_in == NULL_EDGE {
                (*nodes_ptr.add(target as usize)).first_incoming = next_in;
            } else {
                (*edges_ptr.add(prev_in as usize)).next_incoming = next_in;
            }
            if next_in != NULL_EDGE {
                (*edges_ptr.add(next_in as usize)).prev_incoming = prev_in;
            }
            let data = std::ptr::read(&(*edges_ptr.add(idx)).data);
            *gens_ptr.add(idx) = (*gens_ptr.add(idx)).wrapping_add(1);
            self.free_edges.push(edge);
            data
        }
    }

    #[inline]
    pub fn contains_edge(&self, edge: EdgeIndex) -> bool {
        let idx = edge.index as usize;
        idx < self.edges.len()
            && self.edge_generations[idx] == edge.generation
            && self.edge_generations[idx] != 0
    }

    #[inline]
    pub fn edge(&self, edge: EdgeIndex) -> Option<&E> {
        if !self.contains_edge(edge) { return None; }
        Some(&self.edges[edge.index as usize].data)
    }

    #[inline]
    pub fn edge_mut(&mut self, edge: EdgeIndex) -> Option<&mut E> {
        if !self.contains_edge(edge) { return None; }
        Some(&mut self.edges[edge.index as usize].data)
    }

    #[inline]
    pub fn edge_endpoints(&self, edge: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        if !self.contains_edge(edge) { return None; }
        let e = &self.edges[edge.index as usize];
        let s_gen = self.node_generations[e.source as usize];
        let t_gen = self.node_generations[e.target as usize];
        Some((
            NodeIndex { index: e.source, generation: s_gen },
            NodeIndex { index: e.target, generation: t_gen },
        ))
    }

    #[inline]
    pub fn outgoing_edges(&self, node: NodeIndex) -> Option<OutgoingEdges<'_, E>> {
        if !self.contains_node(node) { return None; }
        let first = self.nodes[node.index as usize].first_outgoing;
        Some(OutgoingEdges { edges: &self.edges, edge_generations: &self.edge_generations, next_edge: first })
    }

    #[inline]
    pub fn incoming_edges(&self, node: NodeIndex) -> Option<IncomingEdges<'_, E>> {
        if !self.contains_node(node) { return None; }
        let first = self.nodes[node.index as usize].first_incoming;
        Some(IncomingEdges { edges: &self.edges, edge_generations: &self.edge_generations, next_edge: first })
    }

    #[inline]
    pub fn successors(&self, node: NodeIndex) -> Option<Successors<'_, E>> {
        if !self.contains_node(node) { return None; }
        let first = self.nodes[node.index as usize].first_outgoing;
        Some(Successors { edges: &self.edges, node_generations: &self.node_generations, next_edge: first })
    }

    #[inline]
    pub fn predecessors(&self, node: NodeIndex) -> Option<Predecessors<'_, E>> {
        if !self.contains_node(node) { return None; }
        let first = self.nodes[node.index as usize].first_incoming;
        Some(Predecessors { edges: &self.edges, node_generations: &self.node_generations, next_edge: first })
    }

    pub(crate) fn outgoing_edges_raw(&self, node: NodeIndex) -> OutgoingEdges<'_, E> {
        OutgoingEdges {
            edges: &self.edges,
            edge_generations: &self.edge_generations,
            next_edge: self.nodes[node.index as usize].first_outgoing,
        }
    }

    pub(crate) fn incoming_edges_raw(&self, node: NodeIndex) -> IncomingEdges<'_, E> {
        IncomingEdges {
            edges: &self.edges,
            edge_generations: &self.edge_generations,
            next_edge: self.nodes[node.index as usize].first_incoming,
        }
    }
}

pub struct OutgoingEdges<'a, E> {
    edges: &'a [EdgeEntry<E>],
    edge_generations: &'a [u32],
    next_edge: u32,
}

pub struct IncomingEdges<'a, E> {
    edges: &'a [EdgeEntry<E>],
    edge_generations: &'a [u32],
    next_edge: u32,
}

pub struct Successors<'a, E> {
    edges: &'a [EdgeEntry<E>],
    node_generations: &'a [u32],
    next_edge: u32,
}

pub struct Predecessors<'a, E> {
    edges: &'a [EdgeEntry<E>],
    node_generations: &'a [u32],
    next_edge: u32,
}

impl<'a, E> Iterator for OutgoingEdges<'a, E> {
    type Item = EdgeIndex;
    fn next(&mut self) -> Option<EdgeIndex> {
        if self.next_edge == NULL_EDGE { return None; }
        let current = self.next_edge;
        let idx = current as usize;
        let generation = self.edge_generations[idx];
        self.next_edge = self.edges[idx].next_outgoing;
        Some(EdgeIndex { index: current, generation })
    }
}

impl<'a, E> Iterator for IncomingEdges<'a, E> {
    type Item = EdgeIndex;
    fn next(&mut self) -> Option<EdgeIndex> {
        if self.next_edge == NULL_EDGE { return None; }
        let current = self.next_edge;
        let idx = current as usize;
        let generation = self.edge_generations[idx];
        self.next_edge = self.edges[idx].next_incoming;
        Some(EdgeIndex { index: current, generation })
    }
}

impl<'a, E> Iterator for Successors<'a, E> {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<NodeIndex> {
        if self.next_edge == NULL_EDGE { return None; }
        let current = self.next_edge;
        let idx = current as usize;
        let target = self.edges[idx].target;
        let generation = self.node_generations[target as usize];
        self.next_edge = self.edges[idx].next_outgoing;
        Some(NodeIndex { index: target, generation })
    }
}

impl<'a, E> Iterator for Predecessors<'a, E> {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<NodeIndex> {
        if self.next_edge == NULL_EDGE { return None; }
        let current = self.next_edge;
        let idx = current as usize;
        let source = self.edges[idx].source;
        let generation = self.node_generations[source as usize];
        self.next_edge = self.edges[idx].next_incoming;
        Some(NodeIndex { index: source, generation })
    }
}

impl<N, E> Drop for Graph<N, E> {
    fn drop(&mut self) {
        // 构建空闲槽位集合，跳过已 drop 的槽位
        let free_node_set: std::collections::HashSet<u32> = self.free_nodes.iter().copied().collect();
        let free_edge_set: std::collections::HashSet<u32> = self.free_edges.iter().copied().collect();
        for (i, edge) in self.edges.iter_mut().enumerate() {
            if !free_edge_set.contains(&(i as u32)) {
                unsafe { std::ptr::drop_in_place(&mut edge.data); }
            }
        }
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if !free_node_set.contains(&(i as u32)) {
                unsafe { std::ptr::drop_in_place(&mut node.data); }
            }
        }
        unsafe {
            self.edges.set_len(0);
            self.nodes.set_len(0);
        }
    }
}

unsafe impl<N: Send, E: Send> Send for Graph<N, E> {}
unsafe impl<N: Sync, E: Sync> Sync for Graph<N, E> {}

impl<N: fmt::Debug, E: fmt::Debug> fmt::Debug for Graph<N, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph {{ nodes: {}, edges: {} }}", self.node_count(), self.edge_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_remove_node() {
        let mut g: Graph<&str, ()> = Graph::new();
        let a = g.add_node("A");
        let _b = g.add_node("B");
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.node(a), Some(&"A"));
        let data = g.remove_node(a).unwrap();
        assert_eq!(data, "A");
        assert_eq!(g.node_count(), 1);
        assert!(!g.contains_node(a));
        assert_eq!(g.node(a), None);
    }

    #[test]
    fn test_node_reuse() {
        let mut g: Graph<i32, ()> = Graph::new();
        let n1 = g.add_node(1);
        g.remove_node(n1);
        let n2 = g.add_node(2);
        assert_eq!(n1.index, n2.index);
        assert_ne!(n1.generation, n2.generation);
        assert!(!g.contains_node(n1));
        assert!(g.contains_node(n2));
        assert_eq!(g.node(n2), Some(&2));
    }

    #[test]
    fn test_add_remove_edge() {
        let mut g: Graph<&str, &str> = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let e = g.add_edge(a, b, "edge").unwrap();
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.edge(e), Some(&"edge"));
        let (src, tgt) = g.edge_endpoints(e).unwrap();
        assert_eq!(src, a);
        assert_eq!(tgt, b);
        let data = g.remove_edge(e).unwrap();
        assert_eq!(data, "edge");
        assert_eq!(g.edge_count(), 0);
        assert!(!g.contains_edge(e));
    }

    #[test]
    fn test_edge_invalid_node() {
        let mut g: Graph<(), ()> = Graph::new();
        let a = g.add_node(());
        let fake = NodeIndex { index: 999, generation: 1 };
        assert!(g.add_edge(a, fake, ()).is_none());
        assert!(g.add_edge(fake, a, ()).is_none());
    }

    #[test]
    fn test_successors_predecessors() {
        let mut g: Graph<&str, i32> = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge(a, b, 1);
        g.add_edge(a, c, 2);
        g.add_edge(b, c, 3);
        let succ: Vec<_> = g.successors(a).unwrap().collect();
        assert_eq!(succ.len(), 2);
        assert!(succ.contains(&b));
        assert!(succ.contains(&c));
        let pred: Vec<_> = g.predecessors(c).unwrap().collect();
        assert_eq!(pred.len(), 2);
        assert!(pred.contains(&a));
        assert!(pred.contains(&b));
    }

    #[test]
    fn test_remove_node_cascades_edges() {
        let mut g: Graph<&str, &str> = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge(a, b, "ab");
        g.add_edge(b, c, "bc");
        g.add_edge(c, a, "ca");
        g.remove_node(b);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
        assert!(!g.contains_node(b));
    }

    #[test]
    fn test_degree() {
        let mut g: Graph<(), ()> = Graph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        g.add_edge(a, b, ());
        g.add_edge(a, c, ());
        g.add_edge(b, c, ());
        assert_eq!(g.degree_out(a), Some(2));
        assert_eq!(g.degree_in(a), Some(0));
        assert_eq!(g.degree_out(b), Some(1));
        assert_eq!(g.degree_in(b), Some(1));
        assert_eq!(g.degree_out(c), Some(0));
        assert_eq!(g.degree_in(c), Some(2));
    }

    #[test]
    fn test_clear() {
        let mut g: Graph<&str, &str> = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        g.add_edge(a, b, "e");
        g.clear();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(!g.contains_node(a));
    }

    #[test]
    fn test_self_loop() {
        let mut g: Graph<&str, &str> = Graph::new();
        let a = g.add_node("A");
        let e = g.add_edge(a, a, "self").unwrap();
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.degree_out(a), Some(1));
        assert_eq!(g.degree_in(a), Some(1));
        let succ: Vec<_> = g.successors(a).unwrap().collect();
        assert_eq!(succ, vec![a]);
        let pred: Vec<_> = g.predecessors(a).unwrap().collect();
        assert_eq!(pred, vec![a]);
        g.remove_edge(e);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_mutate_data() {
        let mut g: Graph<i32, i32> = Graph::new();
        let a = g.add_node(10);
        let e = g.add_edge(a, a, 42).unwrap();
        *g.node_mut(a).unwrap() = 99;
        assert_eq!(g.node(a), Some(&99));
        *g.edge_mut(e).unwrap() = 7;
        assert_eq!(g.edge(e), Some(&7));
    }

    #[test]
    fn test_large_graph() {
        let mut g: Graph<usize, usize> = Graph::with_capacity(1000, 5000);
        let mut nodes = Vec::new();
        for i in 0..1000 {
            nodes.push(g.add_node(i));
        }
        for i in 0..999 {
            g.add_edge(nodes[i], nodes[i + 1], i);
        }
        assert_eq!(g.node_count(), 1000);
        assert_eq!(g.edge_count(), 999);
    }
}