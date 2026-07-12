//! cuda emit — 委托 `crate::cuda::kernels::generate` 生成 CUDA C++ 源码

use crate::spec::*;
use base::Result;

pub fn emit(kernels: &[KernelSpec], arch: GpuArch) -> Result<BackendOutput> {
    crate::cuda::kernels::generate(kernels, arch)
}
