//! cuda emit — stub（由子代理填充）

use crate::spec::*;
use base::Result;

pub fn emit(kernels: &[KernelSpec], arch: GpuArch) -> Result<BackendOutput> {
    crate::cuda::kernels::generate(kernels, arch)
}
