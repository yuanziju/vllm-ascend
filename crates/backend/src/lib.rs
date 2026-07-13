//! backend — 后端代码生成
//!
//! 把 regalloc 产出的 MachineInstr + IR Graph 的张量元信息，
//! 翻译成目标后端（CUDA / Triton / Metal / Ascend CANN）的真实源码。
//!
//! 四个后端各有一个 emit 模块，从 KernelSpec 列表生成源码字符串：
//! - CUDA：`.cu` kernel（Hopper wgmma + TMA / Blackwell tensor memory / Ampere mma.sync）
//! - Triton：`.py` `@triton.jit` kernel（Python DSL，支持 SM90 TMA）
//! - Metal：`.metal` MSL kernel（simdgroup_matrix）
//! - CANN：`.cpp` Ascend 算子（Vector Core + Cube Core）
//!
//! 每个 OpKind（31 个变体）在每个后端都有真实实现，无空缺。

pub mod cann;
pub mod cuda;
pub mod extract;
pub mod metal;
pub mod spec;
pub mod triton;

pub use extract::*;
pub use spec::*;

use base::Graph;
use common::Target;

/// 后端代码生成入口
///
/// 从 IR Graph 提取 KernelSpec 列表，按 target + arch 生成对应后端源码。
pub fn emit(
    graph: &Graph,
    _machine_instrs: &[regalloc::MachineInstr],
    target: Target,
    arch: GpuArch,
) -> base::Result<BackendOutput> {
    let kernels = extract_kernels(graph);
    match target {
        Target::Cuda => cuda::emit(&kernels, arch),
        Target::Npu => cann::emit(&kernels, arch),
        Target::Cpu => {
            // CPU 后端不在本轮范围（用户要求 CUDA/Triton/Metal/CANN）
            // 但不能 panic，返回空输出
            Ok(BackendOutput {
                source: "// CPU backend not implemented in this round\n".to_string(),
                lang: SourceLang::Cpu,
                kernels: vec![],
                arch,
            })
        }
    }
}

/// 生成指定后端的源码（直接指定后端类型，不走 Target 映射）
pub fn emit_for(graph: &Graph, lang: SourceLang, arch: GpuArch) -> base::Result<BackendOutput> {
    let kernels = extract_kernels(graph);
    match lang {
        SourceLang::Cuda => cuda::emit(&kernels, arch),
        SourceLang::Triton => triton::emit(&kernels, arch),
        SourceLang::Metal => metal::emit(&kernels, arch),
        SourceLang::Cann => cann::emit(&kernels, arch),
        SourceLang::Cpu => Ok(BackendOutput {
            source: "// CPU backend not implemented in this round\n".to_string(),
            lang: SourceLang::Cpu,
            kernels: vec![],
            arch,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    /// 构造极简图：x → add(x,x) → out，用于触发 emit 路由
    fn mk_simple_graph() -> Graph {
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
        g
    }

    /// emit() 按 Target::Cuda 路由到 CUDA 后端，源码非空且标记为 Cuda
    #[test]
    fn emit_routes_cuda_target() {
        let g = mk_simple_graph();
        let out = emit(&g, &[], common::Target::Cuda, GpuArch::Ampere80).unwrap();
        assert!(!out.source.is_empty(), "CUDA 源码不应为空");
        assert_eq!(out.lang, SourceLang::Cuda);
        assert!(!out.kernels.is_empty(), "应至少生成一个 kernel");
    }

    /// emit() 按 Target::Npu 路由到 CANN 后端
    #[test]
    fn emit_routes_npu_target() {
        let g = mk_simple_graph();
        let out = emit(&g, &[], common::Target::Npu, GpuArch::Ascend910B1).unwrap();
        assert!(!out.source.is_empty());
        assert_eq!(out.lang, SourceLang::Cann);
    }

    /// emit() CPU target 返回占位输出，不 panic
    #[test]
    fn emit_cpu_target_returns_placeholder() {
        let g = mk_simple_graph();
        let out = emit(&g, &[], common::Target::Cpu, GpuArch::Ampere80).unwrap();
        assert!(out.source.contains("CPU backend not implemented"));
        assert_eq!(out.lang, SourceLang::Cpu);
        assert!(out.kernels.is_empty(), "CPU 占位不应有 kernel");
    }

    /// emit_for() 按 SourceLang 路由到所有四种后端
    #[test]
    fn emit_for_routes_all_langs() {
        let g = mk_simple_graph();
        let arch = GpuArch::Ampere80;

        let cuda = emit_for(&g, SourceLang::Cuda, arch).unwrap();
        assert_eq!(cuda.lang, SourceLang::Cuda);
        assert!(!cuda.source.is_empty());

        let triton = emit_for(&g, SourceLang::Triton, arch).unwrap();
        assert_eq!(triton.lang, SourceLang::Triton);
        assert!(!triton.source.is_empty());

        let metal = emit_for(&g, SourceLang::Metal, arch).unwrap();
        assert_eq!(metal.lang, SourceLang::Metal);
        assert!(!metal.source.is_empty());

        let cann = emit_for(&g, SourceLang::Cann, arch).unwrap();
        assert_eq!(cann.lang, SourceLang::Cann);
        assert!(!cann.source.is_empty());
    }

    /// emit_for() CPU 占位
    #[test]
    fn emit_for_cpu_returns_placeholder() {
        let g = mk_simple_graph();
        let out = emit_for(&g, SourceLang::Cpu, GpuArch::Ampere80).unwrap();
        assert!(out.source.contains("CPU backend not implemented"));
        assert_eq!(out.lang, SourceLang::Cpu);
    }

    /// 空图 emit 不应 panic（虽然可能输出空源码）
    #[test]
    fn emit_empty_graph_does_not_panic() {
        let g = Graph::new("empty");
        let out = emit(&g, &[], common::Target::Cuda, GpuArch::Ampere80).unwrap();
        // 空图应正常返回（无 kernel）
        assert!(out.kernels.is_empty());
    }

    /// emit 应保持 arch 信息透传到 BackendOutput
    #[test]
    fn emit_preserves_arch_info() {
        let g = mk_simple_graph();
        for arch in [GpuArch::Ampere80, GpuArch::Hopper90, GpuArch::Blackwell100] {
            let out = emit(&g, &[], common::Target::Cuda, arch).unwrap();
            assert_eq!(out.arch, arch, "arch 应透传到 BackendOutput");
        }
    }
}
