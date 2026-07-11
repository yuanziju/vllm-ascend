//! triton emit — stub（由子代理填充）

use crate::spec::*;
use base::Result;

pub fn emit(_kernels: &[KernelSpec], _arch: GpuArch) -> Result<BackendOutput> {
    Ok(BackendOutput {
        source: String::new(),
        lang: SourceLang::Triton,
        kernels: vec![],
        arch: GpuArch::Hopper90,
    })
}
