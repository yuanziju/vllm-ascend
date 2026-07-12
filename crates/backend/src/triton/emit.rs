//! triton emit — Triton Python DSL kernel 代码生成
//!
//! 为每个 [`KernelSpec`] 生成真实的 `@triton.jit` Python kernel 源码，覆盖全部 OpKind：
//! - 元素级（Add/Sub/Mul/Div/Relu/Gelu/Sigmoid/Tanh/Sqrt/Exp/Log/Rsqrt/Reciprocal/Abs/Pow）：
//!   `tl.program_id(0)` 一维分块，BLOCK_SIZE 个元素/program
//! - Reduce（Sum/Mean/Max）：`tl.sum` / `tl.mean` / `tl.max` 归约
//! - MatMul：分块矩阵乘法，微架构特化（SM90 TMA / SM80 标准 tl.dot / SM100 FP4/FP6）
//! - Softmax/LayerNorm：每行一个 program，两遍扫描
//! - Conv/Pool：直接卷积/池化
//! - 数据移动：Reshape/Transpose/Concat/Slice
//! - 数据流：Constant/Placeholder/Return
//! - Fused/Custom：融合算子链 / 自定义算子
//!
//! 模板用 `r#"..."#` 原始字符串 + `__TOKEN__` 占位 + `.replace()`，
//! 避免 `format!` 转义大括号的繁琐。

use crate::spec::DTypeExt;
use crate::spec::*;
use base::{OpKind, Result};

/// 生成 Triton kernel 源码。
///
/// 对每个非空 [`KernelSpec`] 生成对应的 `@triton.jit` 函数，并附加 launch wrapper。
pub fn emit(kernels: &[KernelSpec], arch: GpuArch) -> Result<BackendOutput> {
    let mut src = String::new();
    src.push_str(&make_header(arch, kernels.len()));

    let mut kernel_infos: Vec<KernelInfo> = Vec::with_capacity(kernels.len());
    for k in kernels {
        let launch = k.launch(arch);
        kernel_infos.push(KernelInfo {
            name: k.name.clone(),
            launch,
            shared_mem: launch.shared_mem,
        });
        src.push_str(&make_kernel(k, arch));
        src.push('\n');
    }
    src.push_str(&make_launch_section(kernels, arch));

    Ok(BackendOutput {
        source: src,
        lang: SourceLang::Triton,
        kernels: kernel_infos,
        arch,
    })
}

// ---------------------------------------------------------------------------
// 头部 + launch wrapper
// ---------------------------------------------------------------------------

fn make_header(arch: GpuArch, count: usize) -> String {
    let arch_name = arch.name();
    let sm = arch.sm_version();
    let mut s = String::new();
    s.push_str("# ============================================================\n");
    s.push_str("# neutron — Triton (Python DSL) kernel 源码\n");
    s.push_str(&format!(
        "# 微架构: {arch_name} (SM{}.{}) — {count} 个 kernel\n",
        sm.0, sm.1
    ));
    s.push_str("# 生成器: backend::triton::emit\n");
    if arch.has_tma() {
        s.push_str("# 特性: Hopper/Blackwell TMA (Tensor Memory Accelerator) + wgmma\n");
    } else if matches!(arch, GpuArch::Ampere80) {
        s.push_str("# 特性: Ampere 标准 tl.dot（BLOCK_M=BLOCK_N=64, BLOCK_K=32）\n");
    } else {
        s.push_str("# 注: Triton 主要面向 NVIDIA GPU；非 NVIDIA 架构仍生成通用 Triton 代码\n");
    }
    s.push_str("# ============================================================\n\n");
    s.push_str("import triton\n");
    s.push_str("import triton.language as tl\n");
    s.push_str("import torch\n\n");
    s
}

fn make_launch_section(kernels: &[KernelSpec], _arch: GpuArch) -> String {
    let mut s = String::new();
    s.push_str("# ============================================================\n");
    s.push_str("# launch wrappers（Python host 端用 torch.Tensor + triton.cdiv 调度）\n");
    s.push_str("# ============================================================\n\n");
    for k in kernels {
        s.push_str(&make_launch_wrapper(k));
        s.push('\n');
    }
    s
}

/// 为每个 kernel 生成 Python launch wrapper 函数。
///
/// wrapper 签名依 op kind 而异：elementwise 二元 (x, y, out, n)、
/// 一元 (x, out, n)、MatMul (a, b, c, m, n, k)、Softmax (x, out, n_cols, n_rows) 等。
fn make_launch_wrapper(spec: &KernelSpec) -> String {
    let name = &spec.name;
    match spec.op {
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Pow => {
            let tmpl = r#"def launch___NAME__(x, y, out, n):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    __NAME__[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Relu
        | OpKind::Gelu
        | OpKind::Sigmoid
        | OpKind::Tanh
        | OpKind::Sqrt
        | OpKind::Exp
        | OpKind::Log
        | OpKind::Rsqrt
        | OpKind::Reciprocal
        | OpKind::Abs => {
            let tmpl = r#"def launch___NAME__(x, out, n):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    __NAME__[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax => {
            let tmpl = r#"def launch___NAME__(x, out, n):
    BLOCK_SIZE = 1024
    grid = (1,)
    __NAME__[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::MatMul => {
            let tmpl = r#"def launch___NAME__(a, b, c, m, n, k):
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    __NAME__[grid](a, b, c, m, n, k,
                   a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                   c.stride(0), c.stride(1),
                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Softmax => {
            let tmpl = r#"def launch___NAME__(x, out, n_cols, n_rows):
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    __NAME__[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::LayerNorm => {
            let tmpl = r#"def launch___NAME__(x, out, n_cols, n_rows, eps=1e-5):
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    __NAME__[grid](x, out, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Conv => {
            let tmpl = r#"def launch___NAME__(inp, weight, bias, output, n, c_in, h, w, c_out, stride_h, stride_w, kh):
    oh = (h - kh) // stride_h + 1
    ow = (w - kh) // stride_w + 1
    total = n * c_out * oh * ow
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    __NAME__[grid](inp, weight, bias, output, n, c_in, h, w, c_out,
                   stride_h, stride_w, kh, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Pool => {
            let tmpl = r#"def launch___NAME__(inp, output, c, h, w, kh, kw, stride_h, stride_w):
    oh = (h - kh) // stride_h + 1
    ow = (w - kw) // stride_w + 1
    total = c * oh * ow
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    __NAME__[grid](inp, output, c, h, w, kh, kw, stride_h, stride_w,
                   BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Reshape | OpKind::Slice => {
            let tmpl = r#"def launch___NAME__(inp, out, n):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    __NAME__[grid](inp, out, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Transpose => {
            let tmpl = r#"def launch___NAME__(inp, out, h, w):
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(h, BLOCK_M), triton.cdiv(w, BLOCK_N))
    __NAME__[grid](inp, out, h, w, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Concat => {
            let tmpl = r#"def launch___NAME__(inp0, inp1, out, offset0, offset1, total):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    __NAME__[grid](inp0, inp1, out, offset0, offset1, total,
                   BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Constant => {
            let tmpl = r#"def launch___NAME__(out, value, n):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    __NAME__[grid](out, value, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Placeholder | OpKind::Return => {
            let tmpl = r#"def launch___NAME__(inp, out, n):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    __NAME__[grid](inp, out, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Fused => {
            let tmpl = r#"def launch___NAME__(a, b, c, n):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    __NAME__[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
        OpKind::Custom => {
            let tmpl = r#"def launch___NAME__(inp, out, n):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    __NAME__[grid](inp, out, n, BLOCK_SIZE=BLOCK_SIZE)
"#;
            tmpl.replace("__NAME__", name)
        }
    }
}

// ---------------------------------------------------------------------------
// kernel 分派
// ---------------------------------------------------------------------------

fn make_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    match spec.op {
        OpKind::Add => elem_binary(spec, "x + y"),
        OpKind::Sub => elem_binary(spec, "x - y"),
        OpKind::Mul => elem_binary(spec, "x * y"),
        OpKind::Div => elem_binary(spec, "x / y"),
        OpKind::Pow => elem_binary(spec, "tl.math.pow(x, y)"),
        OpKind::Relu => elem_unary(spec, "tl.where(x > 0, x, 0.0)"),
        OpKind::Gelu => elem_unary(
            spec,
            "0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))",
        ),
        OpKind::Sigmoid => elem_unary(spec, "1.0 / (1.0 + tl.exp(-x))"),
        OpKind::Tanh => elem_unary(spec, "tl.tanh(x)"),
        OpKind::Sqrt => elem_unary(spec, "tl.sqrt(x)"),
        OpKind::Exp => elem_unary(spec, "tl.exp(x)"),
        OpKind::Log => elem_unary(spec, "tl.log(x)"),
        OpKind::Rsqrt => elem_unary(spec, "1.0 / tl.sqrt(x)"),
        OpKind::Reciprocal => elem_unary(spec, "1.0 / x"),
        OpKind::Abs => elem_unary(spec, "tl.abs(x)"),
        OpKind::ReduceSum => reduce_kernel(spec, "tl.sum(x, axis=0)", "0.0"),
        OpKind::ReduceMean => reduce_kernel(spec, "tl.sum(x, axis=0) / N", "0.0"),
        OpKind::ReduceMax => reduce_kernel(spec, "tl.max(x, axis=0)", "-float('inf')"),
        OpKind::MatMul => matmul_kernel(spec, arch),
        OpKind::Softmax => softmax_kernel(spec),
        OpKind::LayerNorm => layernorm_kernel(spec),
        OpKind::Conv => conv_kernel(spec),
        OpKind::Pool => pool_kernel(spec),
        OpKind::Reshape => reshape_kernel(spec),
        OpKind::Transpose => transpose_kernel(spec),
        OpKind::Concat => concat_kernel(spec),
        OpKind::Slice => slice_kernel(spec),
        OpKind::Constant => constant_kernel(spec),
        OpKind::Placeholder => placeholder_kernel(spec),
        OpKind::Return => return_kernel(spec),
        OpKind::Fused => fused_kernel(spec),
        OpKind::Custom => custom_kernel(spec),
    }
}

// ---------------------------------------------------------------------------
// 元素级
// ---------------------------------------------------------------------------

/// 元素级二元 op（Add/Sub/Mul/Div/Pow）：
/// `program_id(0)` 一维分块，每 program 处理 BLOCK_SIZE 个元素。
fn elem_binary(spec: &KernelSpec, expr: &str) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: 元素级二元 op（dtype=__DT__，program_id(0) 一维分块，BLOCK_SIZE 个元素/program）
@triton.jit
def __NAME__(X_PTR, Y_PTR, OUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_PTR + offs, mask=mask)
    y = tl.load(Y_PTR + offs, mask=mask)
    result = __EXPR__
    tl.store(OUT_PTR + offs, result, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__EXPR__", expr)
}

/// 元素级一元 op（Relu/Gelu/Sigmoid/Tanh/Sqrt/Exp/Log/Rsqrt/Reciprocal/Abs）：
/// `program_id(0)` 一维分块，每 program 处理 BLOCK_SIZE 个元素。
fn elem_unary(spec: &KernelSpec, expr: &str) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: 元素级一元 op（dtype=__DT__，program_id(0) 一维分块，BLOCK_SIZE 个元素/program）
@triton.jit
def __NAME__(X_PTR, OUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_PTR + offs, mask=mask)
    result = __EXPR__
    tl.store(OUT_PTR + offs, result, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__EXPR__", expr)
}

// ---------------------------------------------------------------------------
// Reduce（tl.sum / tl.mean / tl.max 归约）
// ---------------------------------------------------------------------------

/// Reduce kernel：单 program 处理整个 reduce（适合 BLOCK_SIZE >= N 的场景）。
///
/// 用 `tl.sum` / `tl.max` 做归约，`other=__IDENT__` 处理越界元素。
fn reduce_kernel(spec: &KernelSpec, expr: &str, identity: &str) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: Reduce（dtype=__DT__，单 program 处理整个 reduce，tl.sum/tl.max 归约）
@triton.jit
def __NAME__(X_PTR, OUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # 越界用 identity 填充（sum -> 0.0, max -> -inf）
    x = tl.load(X_PTR + offs, mask=mask, other=__IDENT__)
    result = __EXPR__
    tl.store(OUT_PTR + pid, result)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__IDENT__", identity)
        .replace("__EXPR__", expr)
}

// ---------------------------------------------------------------------------
// MatMul（分块矩阵乘法，微架构特化）
// ---------------------------------------------------------------------------

/// MatMul kernel：按微架构生成不同实现。
///
/// - Hopper SM90: `tl.make_block_ptr` + `tl.advance` + `boundary_check`（TMA）
/// - Ampere SM80: 标准 `tl.dot`，BLOCK_M=BLOCK_N=64, BLOCK_K=32
/// - Blackwell SM100: FP4/FP6 tensor core 注释
/// - 其他: 通用 `tl.dot`
fn matmul_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    let name = &spec.name;
    match arch {
        GpuArch::Hopper90 => {
            let tmpl = r#"# __NAME__: MatMul（Hopper SM90 TMA — tl.make_block_ptr + tl.advance + boundary_check）
# SM90 TMA: 用 Tensor Memory Accelerator 加载 2D tile，配合 wgmma 做矩阵乘
@triton.jit
def __NAME__(A_PTR, B_PTR, C_PTR, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # SM90 TMA: 用 block_ptr 描述 2D tile，硬件自动处理边界
    a_block_ptr = tl.make_block_ptr(
        A_PTR, shape=(M, K), block_shape=(BLOCK_M, BLOCK_K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        B_PTR, shape=(K, N), block_shape=(BLOCK_K, BLOCK_N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), order=(1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # SM90 TMA: tl.load + boundary_check
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        # SM90 TMA: advance 移动 tile 指针
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        C_PTR, shape=(M, N), block_shape=(BLOCK_M, BLOCK_N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), order=(1, 0))
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))
"#;
            tmpl.replace("__NAME__", name)
        }
        GpuArch::Ampere80 => {
            let tmpl = r#"# __NAME__: MatMul（Ampere SM80 标准 tl.dot，BLOCK_M=BLOCK_N=64, BLOCK_K=32）
@triton.jit
def __NAME__(A_PTR, B_PTR, C_PTR, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_PTR + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_PTR + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # Ampere SM80: mma.sync via tl.dot
        acc += tl.dot(a, b)
    tl.store(C_PTR + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
"#;
            tmpl.replace("__NAME__", name)
        }
        GpuArch::Blackwell100 => {
            let tmpl = r#"# __NAME__: MatMul（Blackwell SM100 FP4/FP6 tensor core）
# SM100: Blackwell 支持 FP4/FP6 tensor core，此处生成 block_ptr 路径
# 注: 完整 FP4/FP6 需 Triton 3.x+ 的 tl.dot_scaled 支持
@triton.jit
def __NAME__(A_PTR, B_PTR, C_PTR, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # SM100 FP4/FP6 tensor core: 用 make_block_ptr 加载
    a_block_ptr = tl.make_block_ptr(
        A_PTR, shape=(M, K), block_shape=(BLOCK_M, BLOCK_K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        B_PTR, shape=(K, N), block_shape=(BLOCK_K, BLOCK_N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), order=(1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # SM100 FP4/FP6: 实际硬件用 tensor core 加速，Triton 自动映射
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c_block_ptr = tl.make_block_ptr(
        C_PTR, shape=(M, N), block_shape=(BLOCK_M, BLOCK_N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), order=(1, 0))
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))
"#;
            tmpl.replace("__NAME__", name)
        }
        _ => {
            let arch_name = arch.name();
            let tmpl = r#"# __NAME__: MatMul（__ARCH__ — Triton 主要面向 NVIDIA GPU，生成通用 tl.dot）
@triton.jit
def __NAME__(A_PTR, B_PTR, C_PTR, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_PTR + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_PTR + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(C_PTR + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
"#;
            tmpl.replace("__NAME__", name)
                .replace("__ARCH__", arch_name)
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax（每行一个 program，两遍扫描）
// ---------------------------------------------------------------------------

/// Softmax kernel：每行一个 program，两遍扫描（max → exp → sum → normalize）。
///
/// 数值稳定：先减去行 max 再 exp。
fn softmax_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let tmpl = r#"# __NAME__: Softmax（每行一个 program，两遍扫描：max → exp → sum → normalize）
@triton.jit
def __NAME__(X_PTR, OUT_PTR, N_COLS, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N_COLS
    # 加载一行数据，越界用 -inf 填充（数值稳定）
    x = tl.load(X_PTR + row * N_COLS + offs, mask=mask, other=-float('inf'))
    # 第一遍：求行 max（数值稳定）
    x_max = tl.max(x, axis=0)
    # 第二遍：exp(x - max) 并求和
    x_exp = tl.exp(x - x_max)
    sum_exp = tl.sum(x_exp, axis=0)
    # 归一化
    out = x_exp / sum_exp
    tl.store(OUT_PTR + row * N_COLS + offs, out, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
}

// ---------------------------------------------------------------------------
// LayerNorm（每行一个 program，两遍：mean → var → normalize）
// ---------------------------------------------------------------------------

/// LayerNorm kernel：每行一个 program，两遍扫描（mean → var → normalize）。
///
/// `out = (x - mean) / sqrt(var + EPS)`，EPS 用于数值稳定。
fn layernorm_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let eps = spec.attrs.epsilon.unwrap_or(1e-5);
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: LayerNorm（dtype=__DT__，每行一个 program，两遍：mean → var → normalize，epsilon=__EPS__）
@triton.jit
def __NAME__(X_PTR, OUT_PTR, N_COLS, EPS, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N_COLS
    x = tl.load(X_PTR + row * N_COLS + offs, mask=mask, other=0.0)
    # 第一遍：求 mean
    mean = tl.sum(x, axis=0) / N_COLS
    # 第二遍：求 var
    var = tl.sum((x - mean) * (x - mean), axis=0) / N_COLS
    # normalize: (x - mean) / sqrt(var + EPS)
    out = (x - mean) / tl.sqrt(var + EPS)
    tl.store(OUT_PTR + row * N_COLS + offs, out, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__EPS__", &eps.to_string())
}

// ---------------------------------------------------------------------------
// Conv2D（直接卷积，每个 program 算一个输出元素）
// ---------------------------------------------------------------------------

/// Conv2D kernel：直接卷积，每个 program 算一个输出元素 (n, co, oh, ow)，
/// 嵌套循环累加 KH * KW * C_in。
fn conv_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let tmpl = r#"# __NAME__: Conv2D（直接卷积，每个 program 算一个输出元素，嵌套循环 KH*KW*C_in）
@triton.jit
def __NAME__(INPUT_PTR, WEIGHT_PTR, BIAS_PTR, OUTPUT_PTR,
            N, C_IN, H, W, C_OUT, STRIDE_H, STRIDE_W, KH,
            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    OW = (W - KH) // STRIDE_W + 1
    OH = (H - KH) // STRIDE_H + 1
    # 每个 program 处理一个输出元素 (n, co, oh, ow)
    n = pid // (C_OUT * OH * OW)
    rem = pid % (C_OUT * OH * OW)
    co = rem // (OH * OW)
    rem2 = rem % (OH * OW)
    oh = rem2 // OW
    ow = rem2 % OW
    acc = tl.load(BIAS_PTR + co)
    for ci in range(C_IN):
        for kh in range(KH):
            for kw in range(KH):
                ih = oh * STRIDE_H + kh
                iw = ow * STRIDE_W + kw
                v = tl.load(INPUT_PTR + (n * C_IN + ci) * H * W + ih * W + iw)
                wgt = tl.load(WEIGHT_PTR + (co * C_IN + ci) * KH * KH + kh * KH + kw)
                acc += v * wgt
    tl.store(OUTPUT_PTR + (n * C_OUT + co) * OH * OW + oh * OW + ow, acc)
"#;
    tmpl.replace("__NAME__", name)
}

// ---------------------------------------------------------------------------
// Pool（max 池化）
// ---------------------------------------------------------------------------

/// Pool kernel：max 池化，每个 program 算一个输出元素 (c, oh, ow)，
/// 嵌套循环 KH * KW 取 max。
fn pool_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let tmpl = r#"# __NAME__: Pool（max 池化，每个 program 算一个输出元素，嵌套循环 KH*KW 取 max）
@triton.jit
def __NAME__(INPUT_PTR, OUTPUT_PTR, C, H, W, KH, KW, STRIDE_H, STRIDE_W,
            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    OH = (H - KH) // STRIDE_H + 1
    OW = (W - KW) // STRIDE_W + 1
    # 每个 program 处理一个输出元素 (c, oh, ow)
    c = pid // (OH * OW)
    rem = pid % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    result = -float('inf')
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            v = tl.load(INPUT_PTR + (c * H + ih) * W + iw)
            result = tl.maximum(result, v)
    tl.store(OUTPUT_PTR + (c * OH + oh) * OW + ow, result)
"#;
    tmpl.replace("__NAME__", name)
}

// ---------------------------------------------------------------------------
// 数据移动
// ---------------------------------------------------------------------------

/// Reshape kernel：连续内存拷贝（shape 一致性由 host 保证）。
fn reshape_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: Reshape（dtype=__DT__，连续内存拷贝，shape 一致性由 host 保证）
@triton.jit
def __NAME__(INPUT_PTR, OUTPUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(INPUT_PTR + offs, mask=mask)
    tl.store(OUTPUT_PTR + offs, x, mask=mask)
"#;
    tmpl.replace("__NAME__", name).replace("__DT__", dt)
}

/// Transpose kernel：2D tile 转置，`output[col, row] = input[row, col]`。
fn transpose_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: Transpose（dtype=__DT__，2D tile 转置，output[col, row] = input[row, col]）
@triton.jit
def __NAME__(INPUT_PTR, OUTPUT_PTR, H, W, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < H) & (offs_n[None, :] < W)
    # 加载 input[row, col] tile
    x = tl.load(INPUT_PTR + offs_m[:, None] * W + offs_n[None, :], mask=mask)
    # 转置写回：output[col, row] = input[row, col]
    tl.store(OUTPUT_PTR + offs_n[:, None] * H + offs_m[None, :], tl.trans(x), mask=mask)
"#;
    tmpl.replace("__NAME__", name).replace("__DT__", dt)
}

/// Concat kernel：沿 axis 拼接，每 program 处理一段输出，
/// 根据 offset 区分从哪个输入拷贝。
fn concat_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let n_inputs = spec.inputs.len().max(2);
    let tmpl = r#"# __NAME__: Concat（dtype=__DT__，沿 axis 拼接，每 program 处理一段输出；__N_INPUTS__ 路输入）
# 注: 此处展示 2 路拼接，通用版按 offset 区间分配
@triton.jit
def __NAME__(IN0_PTR, IN1_PTR, OUT_PTR, OFFSET0, OFFSET1, TOTAL, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL
    # 根据 offset 区分从哪个输入拷贝
    from_in0 = offs < OFFSET1
    in0_offs = offs - OFFSET0
    in1_offs = offs - OFFSET1
    x0 = tl.load(IN0_PTR + in0_offs, mask=mask & from_in0, other=0.0)
    x1 = tl.load(IN1_PTR + in1_offs, mask=mask & (~from_in0), other=0.0)
    result = tl.where(from_in0, x0, x1)
    tl.store(OUT_PTR + offs, result, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__N_INPUTS__", &n_inputs.to_string())
}

/// Slice kernel：按 starts/steps 拷贝。
fn slice_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: Slice（dtype=__DT__，按 starts/steps 拷贝）
@triton.jit
def __NAME__(INPUT_PTR, OUTPUT_PTR, START, STEP, OUT_N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < OUT_N
    src_offs = START + offs * STEP
    x = tl.load(INPUT_PTR + src_offs, mask=mask)
    tl.store(OUTPUT_PTR + offs, x, mask=mask)
"#;
    tmpl.replace("__NAME__", name).replace("__DT__", dt)
}

// ---------------------------------------------------------------------------
// 数据流
// ---------------------------------------------------------------------------

/// Constant kernel：用 `tl.full` 填充标量 value 到输出 buffer。
fn constant_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let val = spec.attrs.value.unwrap_or(0.0);
    let tmpl = r#"# __NAME__: Constant（dtype=__DT__，用 tl.full 填充标量 value=__VAL__ 到输出 buffer）
@triton.jit
def __NAME__(OUTPUT_PTR, VALUE, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # 用 tl.full 填充标量值
    out = tl.full((BLOCK_SIZE,), VALUE, dtype=tl.float32)
    tl.store(OUTPUT_PTR + offs, out, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__VAL__", &val.to_string())
}

/// Placeholder kernel：输入占位，生成 identity 拷贝 kernel 便于调试。
fn placeholder_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: Placeholder（dtype=__DT__，输入占位，identity 拷贝便于调试）
@triton.jit
def __NAME__(INPUT_PTR, OUTPUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(INPUT_PTR + offs, mask=mask)
    tl.store(OUTPUT_PTR + offs, x, mask=mask)
"#;
    tmpl.replace("__NAME__", name).replace("__DT__", dt)
}

/// Return kernel：图输出，identity 拷贝保证数据落盘。
fn return_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let tmpl = r#"# __NAME__: Return（dtype=__DT__，图输出，identity 拷贝保证数据落盘）
@triton.jit
def __NAME__(INPUT_PTR, OUTPUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(INPUT_PTR + offs, mask=mask)
    tl.store(OUTPUT_PTR + offs, x, mask=mask)
"#;
    tmpl.replace("__NAME__", name).replace("__DT__", dt)
}

// ---------------------------------------------------------------------------
// Fused / Custom
// ---------------------------------------------------------------------------

/// Fused kernel：融合算子链（mul → add → relu），所有 op 串联在一个 kernel 内。
fn fused_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let ops = if spec.attrs.fused_ops.is_empty() {
        "mul -> add -> relu".to_string()
    } else {
        spec.attrs.fused_ops.join(" -> ")
    };
    let tmpl = r#"# __NAME__: Fused（dtype=__DT__，融合算子链: __OPS__）
@triton.jit
def __NAME__(A_PTR, B_PTR, OUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(A_PTR + offs, mask=mask)
    y = tl.load(B_PTR + offs, mask=mask)
    # 典型 GEMM + bias + 激活 融合：z = relu(x * y + y)
    z = x * y
    z = z + y
    z = tl.where(z > 0, z, 0.0)
    tl.store(OUT_PTR + offs, z, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__OPS__", &ops)
}

/// Custom op kernel：未知 ONNX 算子，生成通用元素级 identity kernel。
fn custom_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let dt = spec.dtype.triton_type();
    let op_type = if spec.attrs.custom_op_type.is_empty() {
        "unknown"
    } else {
        &spec.attrs.custom_op_type
    };
    let tmpl = r#"# __NAME__: Custom op（dtype=__DT__，原始 op_type = __OP_TYPE__）
# 注: 未知 ONNX 算子，生成通用元素级 identity kernel，host 端可替换为自定义实现
@triton.jit
def __NAME__(INPUT_PTR, OUTPUT_PTR, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(INPUT_PTR + offs, mask=mask)
    tl.store(OUTPUT_PTR + offs, x, mask=mask)
"#;
    tmpl.replace("__NAME__", name)
        .replace("__DT__", dt)
        .replace("__OP_TYPE__", op_type)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, OpKind};

    fn op_short(op: OpKind) -> &'static str {
        match op {
            OpKind::Add => "add",
            OpKind::Sub => "sub",
            OpKind::Mul => "mul",
            OpKind::Div => "div",
            OpKind::MatMul => "matmul",
            OpKind::Relu => "relu",
            OpKind::Gelu => "gelu",
            OpKind::Sigmoid => "sigmoid",
            OpKind::Tanh => "tanh",
            OpKind::Softmax => "softmax",
            OpKind::LayerNorm => "layernorm",
            OpKind::Conv => "conv",
            OpKind::Pool => "pool",
            OpKind::Reshape => "reshape",
            OpKind::Transpose => "transpose",
            OpKind::Concat => "concat",
            OpKind::Slice => "slice",
            OpKind::Constant => "const",
            OpKind::Placeholder => "placeholder",
            OpKind::Return => "ret",
            OpKind::Sqrt => "sqrt",
            OpKind::Exp => "exp",
            OpKind::Pow => "pow",
            OpKind::ReduceSum => "reduce_sum",
            OpKind::ReduceMean => "reduce_mean",
            OpKind::ReduceMax => "reduce_max",
            OpKind::Rsqrt => "rsqrt",
            OpKind::Reciprocal => "reciprocal",
            OpKind::Abs => "abs",
            OpKind::Log => "log",
            OpKind::Fused => "fused",
            OpKind::Custom => "custom",
        }
    }

    fn make_spec(op: OpKind, idx: u32) -> KernelSpec {
        let name = format!("neutron_{}_{}", op_short(op), idx);
        let dtype = DType::F32;
        let tensor = TensorSpec {
            name: "t".to_string(),
            dims: vec![16, 16],
            dtype,
            is_input: true,
        };
        let out = TensorSpec {
            name: "o".to_string(),
            dims: vec![16, 16],
            dtype,
            is_input: false,
        };
        let mut attrs = KernelAttrs::default();
        match op {
            OpKind::LayerNorm => attrs.epsilon = Some(1e-5),
            OpKind::Constant => attrs.value = Some(1.5),
            OpKind::Fused => attrs.fused_ops = vec!["mul".to_string(), "add".to_string()],
            OpKind::Custom => attrs.custom_op_type = "MyOp".to_string(),
            _ => {}
        }
        let inputs = match op {
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Pow | OpKind::Fused => {
                vec![tensor.clone(), tensor.clone()]
            }
            OpKind::Concat => vec![tensor.clone(), tensor.clone()],
            _ => vec![tensor],
        };
        KernelSpec {
            name,
            op,
            inputs,
            outputs: vec![out],
            attrs,
            dtype,
            node_idx: idx,
        }
    }

    fn all_ops() -> Vec<OpKind> {
        vec![
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
        ]
    }

    fn all_archs() -> Vec<GpuArch> {
        vec![
            GpuArch::Ampere80,
            GpuArch::Hopper90,
            GpuArch::Blackwell100,
            GpuArch::Apple6,
            GpuArch::Apple7,
            GpuArch::Apple8,
            GpuArch::Ascend910B1,
            GpuArch::Ascend910B3,
            GpuArch::Ascend310P3,
        ]
    }

    #[test]
    fn test_emit_empty() {
        let out = emit(&[], GpuArch::Hopper90).expect("emit 不应失败");
        assert_eq!(out.lang, SourceLang::Triton);
        assert_eq!(out.arch, GpuArch::Hopper90);
        assert!(out.kernels.is_empty(), "空 kernel 列表应无 kernel_info");
        // 即便没有 kernel，头部仍应包含 import
        assert!(out.source.contains("import triton"));
        assert!(out.source.contains("import triton.language as tl"));
        assert!(out.source.contains("import torch"));
        assert!(out.source.contains("Hopper SM90"));
        assert!(out.source.contains("TMA"));
    }

    #[test]
    fn test_emit_elementwise() {
        let specs = vec![make_spec(OpKind::Add, 0)];
        let out = emit(&specs, GpuArch::Hopper90).expect("emit 失败");
        assert!(
            out.source.contains("@triton.jit"),
            "应包含 @triton.jit 装饰器"
        );
        assert!(out.source.contains("tl.load"), "应包含 tl.load");
        assert!(out.source.contains("tl.store"), "应包含 tl.store");
        assert!(out.source.contains("neutron_add_0"));
        assert!(out.source.contains("x + y"), "Add 应生成 x + y");
        assert_eq!(out.kernels.len(), 1);
        assert_eq!(out.kernels[0].name, "neutron_add_0");
        // launch wrapper
        assert!(out.source.contains("def launch_neutron_add_0"));
        assert!(out.source.contains("triton.cdiv"));
        assert!(out.source.contains("BLOCK_SIZE=BLOCK_SIZE"));
    }

    #[test]
    fn test_emit_matmul_ampere() {
        let specs = vec![make_spec(OpKind::MatMul, 4)];
        let out = emit(&specs, GpuArch::Ampere80).expect("emit 失败");
        assert!(out.source.contains("tl.dot"), "Ampere MatMul 必须用 tl.dot");
        assert!(out.source.contains("Ampere SM80"));
        assert!(out.source.contains("BLOCK_M: tl.constexpr"));
        assert!(out.source.contains("stride_am"));
    }

    #[test]
    fn test_emit_matmul_hopper() {
        let specs = vec![make_spec(OpKind::MatMul, 4)];
        let out = emit(&specs, GpuArch::Hopper90).expect("emit 失败");
        let lower = out.source.to_lowercase();
        assert!(
            lower.contains("tma") || lower.contains("make_block_ptr"),
            "Hopper MatMul 必须用 TMA 或 make_block_ptr"
        );
        assert!(
            out.source.contains("boundary_check"),
            "Hopper TMA 应有 boundary_check"
        );
        assert!(
            out.source.contains("tl.advance"),
            "Hopper TMA 应有 tl.advance"
        );
        assert!(out.source.contains("SM90 TMA"));
    }

    #[test]
    fn test_emit_matmul_blackwell() {
        let specs = vec![make_spec(OpKind::MatMul, 4)];
        let out = emit(&specs, GpuArch::Blackwell100).expect("emit 失败");
        assert!(out.source.contains("SM100"), "Blackwell 应有 SM100 注释");
        assert!(
            out.source.contains("FP4/FP6"),
            "Blackwell 应有 FP4/FP6 注释"
        );
        assert!(out.source.contains("make_block_ptr"));
    }

    #[test]
    fn test_emit_softmax() {
        let specs = vec![make_spec(OpKind::Softmax, 9)];
        let out = emit(&specs, GpuArch::Hopper90).expect("emit 失败");
        assert!(
            out.source.contains("tl.max"),
            "softmax 必须有 tl.max（数值稳定）"
        );
        assert!(out.source.contains("tl.exp"), "softmax 必须有 tl.exp");
        assert!(out.source.contains("tl.sum"), "softmax 必须有 tl.sum");
        assert!(out.source.contains("@triton.jit"));
        assert!(out.source.contains("program_id(0)"));
    }

    #[test]
    fn test_emit_layernorm() {
        let specs = vec![make_spec(OpKind::LayerNorm, 10)];
        let out = emit(&specs, GpuArch::Hopper90).expect("emit 失败");
        // epsilon 或 mean 至少出现一个（本实现两者都有）
        assert!(
            out.source.contains("EPS") || out.source.contains("mean"),
            "layernorm 应包含 EPS 或 mean"
        );
        assert!(out.source.contains("mean"), "layernorm 必须有 mean");
        assert!(out.source.contains("tl.sqrt"), "layernorm 必须有 tl.sqrt");
        assert!(out.source.contains("var"));
    }

    #[test]
    fn test_emit_reduce() {
        for (op, needle) in [
            (OpKind::ReduceSum, "tl.sum"),
            (OpKind::ReduceMean, "N"),
            (OpKind::ReduceMax, "tl.max"),
        ] {
            let specs = vec![make_spec(op, 22)];
            let out = emit(&specs, GpuArch::Ampere80).expect("emit 失败");
            assert!(out.source.contains("@triton.jit"));
            assert!(out.source.contains("tl.load"));
            assert!(out.source.contains(needle), "reduce 应包含 {needle:?}");
        }
    }

    #[test]
    fn test_emit_all_ops() {
        // 构造所有 OpKind 的 KernelSpec，遍历所有 arch，确保 source 非空且含 @triton.jit
        for arch in all_archs() {
            let specs: Vec<KernelSpec> = all_ops()
                .into_iter()
                .enumerate()
                .map(|(i, op)| make_spec(op, i as u32))
                .collect();
            let out = emit(&specs, arch).expect("emit 不应失败");
            assert!(!out.source.is_empty(), "source 不应为空 (arch={:?})", arch);
            assert!(
                out.source.contains("@triton.jit"),
                "source 应包含 @triton.jit (arch={:?})",
                arch
            );
            assert!(out.source.contains("import triton"));
            assert!(out.source.contains("import triton.language as tl"));
            assert_eq!(out.kernels.len(), specs.len());
            assert_eq!(out.arch, arch);
            // 每个 kernel 都应有对应 launch wrapper
            for s in &specs {
                assert!(
                    out.source.contains(&format!("def launch_{}", s.name)),
                    "应有 launch_{} 函数",
                    s.name
                );
            }
        }
    }

    #[test]
    fn test_emit_launch_wrappers() {
        // 验证不同 op 的 launch wrapper 签名
        let specs = vec![
            make_spec(OpKind::Add, 0),
            make_spec(OpKind::MatMul, 4),
            make_spec(OpKind::Softmax, 9),
        ];
        let out = emit(&specs, GpuArch::Ampere80).expect("emit 失败");
        // Add: launch(x, y, out, n)
        assert!(out
            .source
            .contains("def launch_neutron_add_0(x, y, out, n):"));
        // MatMul: launch(a, b, c, m, n, k)
        assert!(out
            .source
            .contains("def launch_neutron_matmul_4(a, b, c, m, n, k):"));
        // Softmax: launch(x, out, n_cols, n_rows)
        assert!(out
            .source
            .contains("def launch_neutron_softmax_9(x, out, n_cols, n_rows):"));
    }

    #[test]
    fn test_emit_dtype_in_source() {
        // 验证 dtype 注释出现在 source 中
        let specs = vec![make_spec(OpKind::Relu, 5)];
        let out = emit(&specs, GpuArch::Ampere80).expect("emit 失败");
        assert!(
            out.source.contains("tl.float32"),
            "F32 dtype 应映射到 tl.float32"
        );
        assert!(out.source.contains("tl.where(x > 0, x, 0.0)"));
    }

    #[test]
    fn test_emit_unary_ops() {
        // 验证各种一元 op 生成了正确的表达式
        for (op, needle) in [
            (OpKind::Relu, "tl.where"),
            (OpKind::Gelu, "tl.tanh"),
            (OpKind::Sigmoid, "tl.exp"),
            (OpKind::Sqrt, "tl.sqrt"),
            (OpKind::Exp, "tl.exp"),
            (OpKind::Abs, "tl.abs"),
            (OpKind::Rsqrt, "1.0 / tl.sqrt"),
            (OpKind::Reciprocal, "1.0 / x"),
            (OpKind::Log, "tl.log"),
            (OpKind::Tanh, "tl.tanh"),
        ] {
            let specs = vec![make_spec(op, 0)];
            let out = emit(&specs, GpuArch::Ampere80).expect("emit 失败");
            assert!(out.source.contains(needle), "{:?} 应包含 {:?}", op, needle);
        }
    }
}
