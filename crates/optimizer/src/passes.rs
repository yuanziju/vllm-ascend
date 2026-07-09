//! passes — 基础 pass（DCE/Verify）

use base::{Graph, NeutronError, Pass, PassContext, Result};
use std::collections::HashSet;

/// 死代码消除：从 outputs 反向 BFS，删除不可达节点，物理重建图
pub struct DeadCodeElim;

impl Pass for DeadCodeElim {
    fn name(&self) -> &str {
        "dce"
    }
    fn run(&mut self, graph: &mut Graph, ctx: &mut PassContext) -> Result<()> {
        let reachable = compute_reachable(graph)?;
        let remove: HashSet<base::NodeId> = graph
            .node_ids()
            .filter(|id| !reachable.contains(id))
            .collect();
        if remove.is_empty() {
            ctx.inc("dce_before");
            return Ok(());
        }
        let (new_graph, _, _) = graph.compact(&remove);
        *graph = new_graph;
        ctx.inc("dce_removed");
        Ok(())
    }
}

/// 通用图验证 pass
pub struct Verify;

impl Pass for Verify {
    fn name(&self) -> &str {
        "verify"
    }
    fn run(&mut self, graph: &mut Graph, _ctx: &mut PassContext) -> Result<()> {
        for id in graph.node_ids() {
            let n = graph.node(id)?;
            for vin in n.inputs() {
                if graph.value(*vin).is_err() {
                    return Err(NeutronError::Opt(format!(
                        "节点 {} 输入 value {} 不存在",
                        n.id, vin
                    )));
                }
            }
        }
        Ok(())
    }
}

fn compute_reachable(graph: &Graph) -> Result<HashSet<base::NodeId>> {
    let mut reachable = HashSet::new();
    let mut stack: Vec<base::ValueId> = graph.outputs().to_vec();
    while let Some(vid) = stack.pop() {
        let v = graph.value(vid)?;
        let def = v.def_node();
        if def == u32::MAX {
            continue;
        }
        if reachable.insert(def) {
            let n = graph.node(def)?;
            stack.extend(n.inputs().iter().copied());
        }
    }
    Ok(reachable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    fn build_graph_with_dead_code() -> Graph {
        let mut g = Graph::new("test");
        let in0 = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("in0"),
        );
        let in1 = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("in1"),
        );
        let add = g.add_node(OpKind::Add);
        let out0 = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("out0"),
            add,
        );
        g.raw.set_node_inputs(add, &[in0, in0]);
        g.raw.set_node_outputs(add, &[out0]);
        g.mark_output(out0);
        let relu = g.add_node(OpKind::Relu);
        let dead_out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("dead"),
            relu,
        );
        g.raw.set_node_inputs(relu, &[in1]);
        g.raw.set_node_outputs(relu, &[dead_out]);
        g
    }

    #[test]
    fn dce_removes_dead_code() {
        let mut g = build_graph_with_dead_code();
        assert_eq!(g.node_count(), 2);
        let mut dce = DeadCodeElim;
        let mut ctx = PassContext::default();
        dce.run(&mut g, &mut ctx).unwrap();
        assert_eq!(g.node_count(), 1);
        let n = g.node(0).unwrap();
        assert_eq!(n.kind, OpKind::Add);
    }
}
