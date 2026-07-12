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
