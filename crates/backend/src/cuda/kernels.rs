//! cuda kernels — stub（由子代理填充）

use crate::spec::*;
use base::Result;

pub fn generate(_kernels: &[KernelSpec], _arch: GpuArch) -> Result<BackendOutput> {
    Ok(BackendOutput {
        source: String::new(),
        lang: SourceLang::Cuda,
        kernels: vec![],
        arch: GpuArch::Hopper90,
    })
}
