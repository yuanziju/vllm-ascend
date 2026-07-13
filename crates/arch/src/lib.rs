//! arch — 架构描述 + lowering + 目标特定优化

pub mod cuda;
pub mod lowering;
pub mod npu;

use base::{Graph, Result};

/// 架构图（lowering 后）
#[derive(Debug, Default)]
pub struct ArchGraph {
    pub ops: Vec<ArchOp>,
    pub target: common::Target,
}

/// 架构操作（lowering 后的算子）
#[derive(Debug, Clone)]
pub enum ArchOp {
    KernelCall(String),
    Load,
    Store,
}

impl ArchGraph {
    pub fn new(target: common::Target) -> Self {
        Self {
            ops: Vec::new(),
            target,
        }
    }
    pub fn add(&mut self, op: ArchOp) {
        self.ops.push(op);
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// lowering 入口
pub fn lower(graph: &Graph, target: common::Target) -> Result<ArchGraph> {
    lowering::lower(graph, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};
    use common::Target;

    /// 构造简单图，验证 lower() 能把每个 OpKind 映射成对应的 KernelCall
    #[test]
    fn lower_basic() {
        let mut g = Graph::new("test");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![2, 3],
        };
        let x = g.add_input(ty.clone(), Some("x"));
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(ty, Some("out"), add);
        g.storage.set_node_inputs(add, &[x, x]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);

        let ag = lower(&g, Target::Cuda).unwrap();
        assert_eq!(ag.len(), 1, "应有 1 个 ArchOp");
        assert!(matches!(ag.ops[0], ArchOp::KernelCall(ref s) if s == "add"));
        assert_eq!(ag.target, Target::Cuda);
    }

    /// 空图 lower 不 panic
    #[test]
    fn lower_empty_graph() {
        let g = Graph::new("empty");
        let ag = lower(&g, Target::Npu).unwrap();
        assert!(ag.is_empty());
        assert_eq!(ag.target, Target::Npu);
    }

    /// lower 覆盖多个 OpKind 变体，每个都生成对应 KernelCall
    #[test]
    fn lower_multi_op_kinds() {
        let mut g = Graph::new("multi");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![4, 8],
        };
        let x = g.add_input(ty.clone(), Some("x"));

        // relu + sqrt + tanh 三连
        let relu = g.add_node(OpKind::Relu);
        let r_out = g.add_value(ty.clone(), Some("r"), relu);
        g.storage.set_node_inputs(relu, &[x]);
        g.storage.set_node_outputs(relu, &[r_out]);

        let sqrt = g.add_node(OpKind::Sqrt);
        let s_out = g.add_value(ty.clone(), Some("s"), sqrt);
        g.storage.set_node_inputs(sqrt, &[r_out]);
        g.storage.set_node_outputs(sqrt, &[s_out]);

        let tanh = g.add_node(OpKind::Tanh);
        let out = g.add_value(ty, Some("out"), tanh);
        g.storage.set_node_inputs(tanh, &[s_out]);
        g.storage.set_node_outputs(tanh, &[out]);
        g.mark_output(out);

        let ag = lower(&g, Target::Cpu).unwrap();
        assert_eq!(ag.len(), 3);
        // relu -> sqrt -> tanh
        assert!(matches!(&ag.ops[0], ArchOp::KernelCall(s) if s == "relu"));
        assert!(matches!(&ag.ops[1], ArchOp::KernelCall(s) if s == "sqrt"));
        assert!(matches!(&ag.ops[2], ArchOp::KernelCall(s) if s == "tanh"));
    }

    /// lowering 显式覆盖所有 OpKind 变体，新增 OpKind 时漏写分支会编译失败
    /// （non-exhaustive match）。此测试验证覆盖完整性：每个 OpKind 都能 lower 成
    /// 非空 kernel 名，不 panic。
    #[test]
    fn lower_all_opkinds_covered() {
        let ops = [
            OpKind::Add,
            OpKind::Sub,
            OpKind::Mul,
            OpKind::Div,
            OpKind::MatMul,
            OpKind::Relu,
            OpKind::Gelu,
            OpKind::Sigmoid,
            OpKind::Tanh,
            OpKind::Softmax,
            OpKind::LayerNorm,
            OpKind::Conv,
            OpKind::Pool,
            OpKind::Reshape,
            OpKind::Transpose,
            OpKind::Concat,
            OpKind::Slice,
            OpKind::Constant,
            OpKind::Placeholder,
            OpKind::Return,
            OpKind::Sqrt,
            OpKind::Exp,
            OpKind::Pow,
            OpKind::ReduceSum,
            OpKind::ReduceMean,
            OpKind::ReduceMax,
            OpKind::Rsqrt,
            OpKind::Reciprocal,
            OpKind::Abs,
            OpKind::Log,
            OpKind::Fused,
            OpKind::Custom,
        ];
        for op in ops {
            let mut g = Graph::new("single");
            let ty = Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            };
            let x = g.add_input(ty.clone(), Some("x"));
            let n = g.add_node(op);
            let out = g.add_value(ty, Some("out"), n);
            g.storage.set_node_inputs(n, &[x, x]);
            g.storage.set_node_outputs(n, &[out]);
            g.mark_output(out);

            let ag = lower(&g, Target::Cuda).unwrap();
            assert_eq!(ag.len(), 1, "OpKind {:?} 应 lower 成 1 个 ArchOp", op);
            match &ag.ops[0] {
                ArchOp::KernelCall(s) => assert!(!s.is_empty(), "OpKind {:?} kernel 名为空", op),
                other => panic!("OpKind {:?} 应为 KernelCall，实际 {:?}", op, other),
            }
        }
    }

    /// ArchGraph 的 add/len/is_empty 基本 API
    #[test]
    fn arch_graph_basic_api() {
        let mut ag = ArchGraph::new(Target::Cuda);
        assert!(ag.is_empty());
        assert_eq!(ag.len(), 0);

        ag.add(ArchOp::Load);
        ag.add(ArchOp::Store);
        ag.add(ArchOp::KernelCall("add".to_string()));
        assert!(!ag.is_empty());
        assert_eq!(ag.len(), 3);
    }

    /// CudaDesc default 值合理
    #[test]
    fn cuda_desc_default_sane() {
        let d = cuda::CudaDesc::default();
        assert!(
            d.compute_capability.0 >= 8,
            "compute_capability major 应 >= 8"
        );
        assert!(d.shared_mem_bytes >= 32 * 1024, "shared_mem 应 >= 32KB");
    }
}
