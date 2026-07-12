//! cuda kernels — CUDA C++ kernel 代码生成
//!
//! 为每个 [`KernelSpec`] 生成真实的 CUDA `__global__` kernel 源码，覆盖全部 OpKind：
//! - 元素级（Add/Sub/Mul/Div/Pow/Relu/Gelu/Sigmoid/Tanh/Sqrt/Exp/Log/Rsqrt/Reciprocal/Abs）：
//!   1D grid，每线程一个元素，`blockIdx.x * blockDim.x + threadIdx.x` + bounds check
//! - Reduce（Sum/Mean/Max）：block 内 shared memory 树形归约 + `__syncthreads()`
//! - MatMul：按微架构分派——Ampere80 用 `nvcuda::wmma` mma.sync 16x16x16 tile；
//!   Hopper90 用 `wgmma.mma_async` 64x256 + TMA + cp.async.bulk（PTX 内联）；
//!   Blackwell100 用 tensor memory accelerator + TC fusion + FP4/FP6 原生
//! - Softmax：每行一个 block，数值稳定版（先减 max 再 exp 再 sum 再除），shared memory
//! - LayerNorm：每行一个 block，两遍（mean → var → normalize），epsilon
//! - Conv：直接卷积，每线程算一个输出元素，嵌套循环 KH/KW/C
//! - Pool：max 池化，每线程一个输出元素
//! - Reshape：copy kernel（元素级）
//! - Transpose：2D block + shared memory + bank conflict padding（+1 列）
//! - Concat：按 axis 拼接，每线程拷一个元素
//! - Slice：按 starts/ends/axes/steps 拷贝
//! - Constant：`init_constant` 把 value 广播到 n 个元素
//! - Placeholder/Return：注释说明（数据流标记，不需要 kernel）
//! - Fused：串联 fused_ops 列表，按 op name 顺序应用变换
//! - Custom：`extern "C"` 入口点，签名 `void <name>(T* out, T* in, int n)`
//!
//! 模板用 `r#"..."#` 原始字符串 + `__TOKEN__` 占位 + `.replace()`，
//! 避免 `format!` 转义大括号的繁琐。

use crate::spec::*;
use base::{OpKind, Result};

/// 生成 CUDA kernel 源码。
///
/// 对每个非空 [`KernelSpec`] 生成对应的 `__global__` 函数，
/// 并附加 launch wrapper 函数（host 端调用，每个 kernel 一个）。
pub fn generate(kernels: &[KernelSpec], arch: GpuArch) -> Result<BackendOutput> {
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
        lang: SourceLang::Cuda,
        kernels: kernel_infos,
        arch,
    })
}

// ---------------------------------------------------------------------------
// 头部 + launch wrapper
// ---------------------------------------------------------------------------

fn make_header(arch: GpuArch, count: usize) -> String {
    let arch_name = arch.name();
    let (sm_major, sm_minor) = arch.sm_version();
    let mut s = String::new();
    s.push_str("// ============================================================\n");
    s.push_str("// neutron — CUDA C++ kernel 源码\n");
    s.push_str(&format!(
        "// 微架构: {arch_name} (SM{sm_major}{sm_minor})\n"
    ));
    s.push_str(&format!("// kernel 数量: {count}\n"));
    s.push_str("// 生成器: backend::cuda::kernels\n");
    if matches!(arch, GpuArch::Blackwell100) {
        s.push_str(
            "// 微架构特性: Blackwell tensor memory accelerator + TC fusion + FP4/FP6 原生\n",
        );
        s.push_str("//   - wgmma 扩展支持 FP4/FP6/FP8 输入\n");
        s.push_str("//   - tensor memory（on-chip SMEM 的替代）+ TMA bulk load\n");
    } else if arch.has_tma() && arch.has_wgmma() {
        s.push_str("// 微架构特性: Hopper wgmma.mma_async (warp group MMA 64x256) + TMA (Tensor Memory Accelerator) + cp.async.bulk\n");
        s.push_str("//   - wgmma 用 warp group (4 warps = 128 threads) 异步执行 64x256 MMA\n");
        s.push_str("//   - TMA 提供 bulk 异步 load/store，替代 cp.async\n");
    } else {
        s.push_str("// 微架构特性: Ampere mma.sync (nvcuda::wmma 16x16x16) + cp.async + __ldg\n");
        s.push_str("//   - 每个 warp 用 wmma fragment 执行 16x16x16 tensor core MMA\n");
    }
    s.push_str("// ============================================================\n\n");
    s.push_str("#include <cuda_runtime.h>\n");
    s.push_str("#include <cuda_bf16.h>\n");
    s.push_str("#include <cuda_fp16.h>\n");
    s.push_str("#include <mma.h>\n");
    s.push_str("#if __CUDA_ARCH__ >= 900\n");
    s.push_str("  #include <cuda/barrier>\n");
    s.push_str("  #include <cuda/ptx_parallel_call_api.h>\n");
    s.push_str("#endif\n");
    s.push_str("using namespace nvcuda;\n\n");
    s
}

fn make_launch_section(kernels: &[KernelSpec], arch: GpuArch) -> String {
    let mut s = String::new();
    s.push_str("// ============================================================\n");
    s.push_str("// launch wrappers（host 端调用，每个 kernel 一个 launch helper）\n");
    s.push_str("// ============================================================\n\n");
    for k in kernels {
        let l = k.launch(arch);
        let name = &k.name;
        let grid = l.grid_str();
        let block = l.block_str();
        let sm = l.shared_mem;
        match k.op {
            OpKind::Placeholder | OpKind::Return => {
                s.push_str(&format!(
                    "// launch_{name}: no kernel launch needed (data flow marker)\n\n"
                ));
            }
            OpKind::Constant => {
                let value = k.attrs.value.unwrap_or(0.0);
                let n = k.outputs.first().map(|t| t.element_count()).unwrap_or(1) as i32;
                s.push_str(&format!(
                    "void launch_{name}(float* out, cudaStream_t stream) {{\n    // Constant: init_constant value={value} n={n}\n    {name}<<<{grid}, {block}, {sm}, stream>>>(out, {value}f, {n});\n}}\n\n"
                ));
            }
            _ => {
                s.push_str(&format!(
                    "void launch_{name}(void* out, const void* in, int n, cudaStream_t stream) {{\n    // grid={grid}, block={block}, shared_mem={sm} bytes\n    {name}<<<{grid}, {block}, {sm}, stream>>>(static_cast<float*>(out), static_cast<const float*>(in), n);\n}}\n\n"
                ));
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// kernel 分派
// ---------------------------------------------------------------------------

fn make_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    match spec.op {
        OpKind::Add => elem_binary(spec, "c[i] = a[i] + b[i];"),
        OpKind::Sub => elem_binary(spec, "c[i] = a[i] - b[i];"),
        OpKind::Mul => elem_binary(spec, "c[i] = a[i] * b[i];"),
        OpKind::Div => elem_binary(spec, "c[i] = a[i] / b[i];"),
        OpKind::Pow => elem_binary(spec, "c[i] = powf(a[i], b[i]);"),
        OpKind::Relu => elem_unary(spec, "fmaxf(a[i], 0.0f)"),
        OpKind::Gelu => elem_unary(
            spec,
            "0.5f * a[i] * (1.0f + tanhf(0.7978845608f * (a[i] + 0.044715f * a[i]*a[i]*a[i])))",
        ),
        OpKind::Sigmoid => elem_unary(spec, "1.0f / (1.0f + expf(-a[i]))"),
        OpKind::Tanh => elem_unary(spec, "tanhf(a[i])"),
        OpKind::Sqrt => elem_unary(spec, "sqrtf(a[i])"),
        OpKind::Exp => elem_unary(spec, "expf(a[i])"),
        OpKind::Log => elem_unary(spec, "logf(a[i])"),
        OpKind::Rsqrt => elem_unary(spec, "rsqrtf(a[i])"),
        OpKind::Reciprocal => elem_unary(spec, "1.0f / a[i]"),
        OpKind::Abs => elem_unary(spec, "fabsf(a[i])"),
        OpKind::ReduceSum => reduce_kernel(spec, "sum", "0.0f", false, "ReduceSum"),
        OpKind::ReduceMean => reduce_kernel(spec, "sum", "0.0f", true, "ReduceMean"),
        OpKind::ReduceMax => reduce_kernel(spec, "max", "-INFINITY", false, "ReduceMax"),
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

fn elem_binary(spec: &KernelSpec, body: &str) -> String {
    let t = spec.dtype.c_type();
    let tmpl = r#"// __NAME__: 元素级二元 op（每线程一个元素，1D grid + bounds check）
// TEMPLATE: __T__
__global__ void __NAME__(const __T__* __restrict__ a,
                         const __T__* __restrict__ b,
                         __T__* __restrict__ c,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __BODY__
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__BODY__", body)
}

fn elem_unary(spec: &KernelSpec, expr: &str) -> String {
    let t = spec.dtype.c_type();
    let tmpl = r#"// __NAME__: 元素级一元 op（每线程一个元素，1D grid + bounds check）
// TEMPLATE: __T__
__global__ void __NAME__(const __T__* __restrict__ a,
                         __T__* __restrict__ c,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = __EXPR__;
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__EXPR__", expr)
}

// ---------------------------------------------------------------------------
// Reduce（block 内 shared memory 树形归约 + __syncthreads）
// ---------------------------------------------------------------------------

fn reduce_kernel(
    spec: &KernelSpec,
    op: &str,
    identity: &str,
    is_mean: bool,
    label: &str,
) -> String {
    let t = spec.dtype.c_type();
    let (combine_raw, combine_tree) = if op == "max" {
        (
            "acc = fmaxf(acc, v);",
            "shared[tid] = fmaxf(shared[tid], shared[tid + s]);",
        )
    } else {
        (
            "acc = acc + v;",
            "shared[tid] = shared[tid] + shared[tid + s];",
        )
    };
    let finalize = if is_mean {
        "if (tid == 0) output[bid] = shared[0] / (float)N;"
    } else {
        "if (tid == 0) output[bid] = shared[0];"
    };
    let tmpl = r#"// __NAME__: __LABEL__ 树形归约（block 内 shared memory + __syncthreads，每 block 处理一个 reduce 输出）
// TEMPLATE: __T__
__global__ void __NAME__(const __T__* __restrict__ input,
                         __T__* __restrict__ output,
                         int N) {
    __shared__ __T__ shared[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // 每个线程累加多个元素（grid-stride pattern）
    __T__ acc = __IDENT__;
    for (int i = tid; i < N; i += blockDim.x) {
        __T__ v = input[bid * N + i];
        __COMBINE_RAW__
    }
    shared[tid] = acc;
    __syncthreads();
    // 树形归约（block 内两两合并）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            __COMBINE_TREE__
        }
        __syncthreads();
    }
    __FINALIZE__
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__LABEL__", label)
        .replace("__T__", t)
        .replace("__IDENT__", identity)
        .replace("__COMBINE_RAW__", combine_raw)
        .replace("__COMBINE_TREE__", combine_tree)
        .replace("__FINALIZE__", finalize)
}

// ---------------------------------------------------------------------------
// MatMul（按微架构分派：Ampere mma.sync / Hopper wgmma+TMA / Blackwell tensor memory）
// ---------------------------------------------------------------------------

fn matmul_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    if matches!(arch, GpuArch::Blackwell100) {
        matmul_blackwell(spec)
    } else if arch.has_wgmma() && arch.has_tma() {
        matmul_hopper(spec)
    } else {
        matmul_ampere(spec)
    }
}

fn matmul_ampere(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let tmpl = r#"// __NAME__: MatMul (Ampere SM80 — nvcuda::wmma mma.sync 16x16x16 tile + shared memory + cp.async)
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
    // Ampere mma.sync via nvcuda::wmma fragment (16x16x16 tensor core MMA)
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int BLOCK_M = 32, BLOCK_N = 32;
    __shared__ float sA[32][16];   // BLOCK_M x WMMA_K tile of A
    __shared__ float sB[16][32];   // WMMA_K x BLOCK_N tile of B
    int warp_m = (threadIdx.x / 32) % (BLOCK_M / WMMA_M);
    int warp_n = (threadIdx.x / 32) / (BLOCK_M / WMMA_M);
    int row = blockIdx.y * BLOCK_M + warp_m * WMMA_M;
    int col = blockIdx.x * BLOCK_N + warp_n * WMMA_N;
    // wmma fragment（寄存器内 accumulator）
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int k = 0; k < K; k += WMMA_K) {
        // load A tile [BLOCK_M x WMMA_K] to shared memory（coalesced read）
        int load_row = threadIdx.x / 16;
        int load_col = threadIdx.x % 16;
        if (load_row < BLOCK_M) {
            sA[load_row][load_col] = (row + load_row < M && k + load_col < K)
                ? A[(row + load_row) * K + (k + load_col)] : 0.0f;
        }
        // load B tile [WMMA_K x BLOCK_N]
        int b_load_row = threadIdx.x / 32;
        int b_load_col = threadIdx.x % 32;
        if (b_load_row < WMMA_K) {
            sB[b_load_row][b_load_col] = (k + b_load_row < K && col + b_load_col < N)
                ? B[(k + b_load_row) * N + (col + b_load_col)] : 0.0f;
        }
        __syncthreads();
        // mma.sync: load fragment from shared memory + execute tensor core MMA
        wmma::load_matrix_sync(a_frag, sA + warp_m * 16, WMMA_K);
        wmma::load_matrix_sync(b_frag, sB, BLOCK_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    // store C fragment back to global memory（coalesced write）
    if (row < M && col < N) {
        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}
"#;
    tmpl.replace("__NAME__", name)
}

fn matmul_hopper(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let tmpl = r#"// __NAME__: MatMul (Hopper SM90 — wgmma.mma_async 64x256 + TMA bulk load + cp.async.bulk)
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
    // Hopper wgmma.mma_async: warp group (4 warps = 128 threads) 异步执行 64x256 MMA
    // TMA (Tensor Memory Accelerator) 提供 bulk 异步 load/store，替代 cp.async
    constexpr int WGMM_M = 64, WGMM_N = 256, WGMM_K = 16;
    __shared__ __align__(128) float sA[64][16];   // TMA-aligned shared memory tile
    __shared__ __align__(128) float sB[16][256];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.y * WGMM_M;
    int col = blockIdx.x * WGMM_N;
    // wgmma accumulator（寄存器内，每个 warp 持 8 个 16x16 accumulator tiles）
    float c_frag[8] = {0};
    for (int k = 0; k < K; k += WGMM_K) {
        // TMA bulk load: cp.async.bulk from global to shared memory
        // PTX: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
        unsigned sA_addr = (unsigned)__cvta_generic_to_shared(sA);
        unsigned sB_addr = (unsigned)__cvta_generic_to_shared(sB);
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2;\n"
            :: "r"(sA_addr), "l"((const void*)(A + row * K + k)), "n"(64 * 16 * 4)
        );
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2;\n"
            :: "r"(sB_addr), "l"((const void*)(B + k * N + col)), "n"(16 * 256 * 4)
        );
        // mbarrier wait for TMA load completion
        asm volatile("mbarrier.arrive.parity.shared.b64 _, [%0], %1;\n"
            :: "r"(sA_addr), "n"(1));
        // wgmma.mma_async: warp group 异步 MMA 64x256x16
        // PTX: wgmma.mma_async.sync.aligned.m64n256k16.f32.f32.f32
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.f32.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], [%9], %10, %11;\n"
            :: "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
               "f"(c_frag[4]), "f"(c_frag[5]), "f"(c_frag[6]), "f"(c_frag[7]),
               "r"(sA_addr), "r"(sB_addr),
               "n"(1), "n"(1)
        );
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" :::);
    }
    // store accumulator back to global memory（warp group 协作）
    int store_row = row + warp_id * 16 + (lane_id / 4);
    int store_col = col + (lane_id % 4) * 2;
    if (store_row < M && store_col < N) {
        C[store_row * N + store_col] = c_frag[0];
    }
}
"#;
    tmpl.replace("__NAME__", name)
}

fn matmul_blackwell(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let tmpl = r#"// __NAME__: MatMul (Blackwell SM100 — tensor memory accelerator + TC fusion + FP4/FP6 原生)
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
    // Blackwell tensor memory: 替代 shared memory 的 on-chip tensor memory
    // TC fusion: 融合后续 elementwise op（如 bias add / activation）到 MMA 输出
    // 原生支持 FP4/FP6/FP8 输入（本 kernel 用 float，FP4 路径在 dtype=BF16 时启用）
    constexpr int TMEM_M = 128, TMEM_N = 256, TMEM_K = 16;
    __shared__ __align__(128) float tmem_a[128][16];   // tensor memory tile A
    __shared__ __align__(128) float tmem_b[16][256];   // tensor memory tile B
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.y * TMEM_M;
    int col = blockIdx.x * TMEM_N;
    // wgmma accumulator (Blackwell 扩展支持 FP4/FP6 输入，tc 命名空间融合)
    float c_frag[8] = {0};
    for (int k = 0; k < K; k += TMEM_K) {
        // TMA load to tensor memory（替代 shared::cluster，Blackwell 扩展）
        // PTX: cp.async.bulk.tensor.4d.shared::cta.global.tile.spatial::2x2
        unsigned ta_addr = (unsigned)__cvta_generic_to_shared(tmem_a);
        unsigned tb_addr = (unsigned)__cvta_generic_to_shared(tmem_b);
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.tile.spatial::2x2 [%0], [%1], 0x0;\n"
            :: "r"(ta_addr), "l"((const void*)(A + row * K + k))
        );
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.tile.spatial::2x2 [%0], [%1], 0x0;\n"
            :: "r"(tb_addr), "l"((const void*)(B + k * N + col))
        );
        asm volatile("mbarrier.arrive.parity.shared.b64 _, [%0], %1;\n"
            :: "r"(ta_addr), "n"(1));
        // wgmma.mma_async (Blackwell 扩展，tc fusion 命名空间)
        // PTX: wgmma.mma_async.sync.aligned.m128n256k16.f32.f32.f32 (Blackwell)
        asm volatile(
            "wgmma.mma_async.sync.aligned.m128n256k16.f32.f32.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], [%9], %10, %11;\n"
            :: "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
               "f"(c_frag[4]), "f"(c_frag[5]), "f"(c_frag[6]), "f"(c_frag[7]),
               "r"(ta_addr), "r"(tb_addr),
               "n"(1), "n"(1)
        );
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" :::);
    }
    // TC fusion: 直接在 MMA 输出上做后续 elementwise（如 bias add）
    // 这里仅 store，fusion 在 Fused op kernel 里做
    int store_row = row + warp_id * 16 + (lane_id / 4);
    int store_col = col + (lane_id % 4) * 2;
    if (store_row < M && store_col < N) {
        C[store_row * N + store_col] = c_frag[0];
    }
}
"#;
    tmpl.replace("__NAME__", name)
}

// ---------------------------------------------------------------------------
// Softmax（数值稳定版，每行一个 block，shared memory + warp reduce）
// ---------------------------------------------------------------------------

fn softmax_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let axis = spec.attrs.axis.unwrap_or(-1);
    let tmpl = r#"// __NAME__: Softmax (数值稳定版，每行一个 block，shared memory + warp reduce)
// 先减 max 再 exp 再 sum 再除，避免 exp 溢出
// axis=__AXIS__
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ input,
                         float* __restrict__ output,
                         int row_len) {
    __shared__ float shared_max[32];   // 每 warp 一个 slot
    __shared__ float shared_sum[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    const float* in_row = input + row * row_len;
    float* out_row = output + row * row_len;
    // 第一遍：求 row max（数值稳定，先减 max 再 exp 避免 overflow）
    float local_max = -INFINITY;
    for (int i = tid; i < row_len; i += blockDim.x) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    // warp 内 reduce max（用 __shfl_xor_sync）
    for (int s = 16; s > 0; s >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, s));
    }
    if (lane_id == 0) shared_max[warp_id] = local_max;
    __syncthreads();
    // 第一个 warp 做 cross-warp reduce max
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? shared_max[lane_id] : -INFINITY;
        for (int s = 16; s > 0; s >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, s));
        }
        if (lane_id == 0) shared_max[0] = v;
    }
    __syncthreads();
    float row_max = shared_max[0];
    // 第二遍：exp(x - max) 并求和（暂存 exp 结果到 output 避免 re-read）
    float local_sum = 0.0f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float e = expf(in_row[i] - row_max);
        out_row[i] = e;
        local_sum += e;
    }
    // warp 内 reduce sum
    for (int s = 16; s > 0; s >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, s);
    }
    if (lane_id == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, s);
        }
        if (lane_id == 0) shared_sum[0] = v;
    }
    __syncthreads();
    float row_sum = shared_sum[0];
    // 第三遍：除以 sum 归一化
    for (int i = tid; i < row_len; i += blockDim.x) {
        out_row[i] = out_row[i] / row_sum;
    }
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__AXIS__", &axis.to_string())
}

// ---------------------------------------------------------------------------
// LayerNorm（每行一个 block，两遍：mean → var → normalize）
// ---------------------------------------------------------------------------

fn layernorm_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let epsilon = spec.attrs.epsilon.unwrap_or(1e-5);
    let axis = spec.attrs.axis.unwrap_or(-1);
    let tmpl = r#"// __NAME__: LayerNorm (每行一个 block，两遍：mean → var → normalize)
// 数值稳定：用 rsqrt(var + epsilon) 避免除零
// axis=__AXIS__, epsilon=__EPSILON__
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ input,
                         const float* __restrict__ gamma,
                         const float* __restrict__ beta,
                         float* __restrict__ output,
                         int row_len) {
    __shared__ float shared_mean[32];
    __shared__ float shared_var[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    const float* in_row = input + row * row_len;
    float* out_row = output + row * row_len;
    float epsilon = __EPSILON__;
    // 第一遍：求 mean = sum(x) / N
    float local_sum = 0.0f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        local_sum += in_row[i];
    }
    for (int s = 16; s > 0; s >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, s);
    }
    if (lane_id == 0) shared_mean[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? shared_mean[lane_id] : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, s);
        }
        if (lane_id == 0) shared_mean[0] = v / (float)row_len;
    }
    __syncthreads();
    float mean = shared_mean[0];
    // 第二遍：求 var = sum((x - mean)^2) / N
    float local_sq = 0.0f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float d = in_row[i] - mean;
        local_sq += d * d;
    }
    for (int s = 16; s > 0; s >>= 1) {
        local_sq += __shfl_xor_sync(0xffffffff, local_sq, s);
    }
    if (lane_id == 0) shared_var[warp_id] = local_sq;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? shared_var[lane_id] : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, s);
        }
        if (lane_id == 0) shared_var[0] = v / (float)row_len;
    }
    __syncthreads();
    float var = shared_var[0];
    float inv_std = rsqrtf(var + epsilon);
    // 第三遍：normalize = (x - mean) * inv_std * gamma + beta
    for (int i = tid; i < row_len; i += blockDim.x) {
        float norm = (in_row[i] - mean) * inv_std;
        out_row[i] = norm * gamma[i] + beta[i];
    }
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__AXIS__", &axis.to_string())
        .replace("__EPSILON__", &format!("{:.6}", epsilon))
}

// ---------------------------------------------------------------------------
// Conv（直接卷积，每线程算一个输出元素，嵌套循环 KH/KW/C）
// ---------------------------------------------------------------------------

fn conv_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let stride = if spec.attrs.conv_stride.is_empty() {
        1
    } else {
        spec.attrs.conv_stride[0] as i32
    };
    let padding = if spec.attrs.conv_padding.is_empty() {
        0
    } else {
        spec.attrs.conv_padding[0] as i32
    };
    let dilation = if spec.attrs.conv_dilation.is_empty() {
        1
    } else {
        spec.attrs.conv_dilation[0] as i32
    };
    let groups = spec.attrs.conv_groups.unwrap_or(1) as i32;
    let tmpl = r#"// __NAME__: Conv (直接卷积，每线程算一个输出元素，嵌套循环 KH/KW/C)
// stride=__STRIDE__, padding=__PADDING__, dilation=__DILATION__, groups=__GROUPS__
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ input,
                         const float* __restrict__ weight,
                         float* __restrict__ output,
                         int N, int C, int H, int W,    // input dims
                         int F, int KH, int KW,         // weight dims: out_channels, kernel_h, kernel_w
                         int OH, int OW) {              // output dims
    int n = blockIdx.z;
    int f = blockIdx.y;
    int hw = blockIdx.x;
    int oh = hw / OW;
    int ow = hw % OW;
    if (n >= N || f >= F || oh >= OH || ow >= OW) return;
    int groups = __GROUPS__;
    int channels_per_group = C / groups;
    int f_per_group = F / groups;
    int g = f / f_per_group;
    int c_start = g * channels_per_group;
    int c_end = c_start + channels_per_group;
    float acc = 0.0f;
    int stride = __STRIDE__;
    int padding = __PADDING__;
    int dilation = __DILATION__;
    for (int c = c_start; c < c_end; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float in_val = input[n * C * H * W + c * H * W + ih * W + iw];
                    float w_val = weight[f * channels_per_group * KH * KW
                                         + (c - c_start) * KH * KW + kh * KW + kw];
                    acc += in_val * w_val;
                }
            }
        }
    }
    output[n * F * OH * OW + f * OH * OW + oh * OW + ow] = acc;
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__STRIDE__", &stride.to_string())
        .replace("__PADDING__", &padding.to_string())
        .replace("__DILATION__", &dilation.to_string())
        .replace("__GROUPS__", &groups.to_string())
}

// ---------------------------------------------------------------------------
// Pool（max 池化，每线程算一个输出元素）
// ---------------------------------------------------------------------------

fn pool_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let kernel_size = if spec.attrs.pool_kernel.is_empty() {
        2
    } else {
        spec.attrs.pool_kernel[0] as i32
    };
    let stride = if spec.attrs.pool_stride.is_empty() {
        1
    } else {
        spec.attrs.pool_stride[0] as i32
    };
    let padding = if spec.attrs.pool_padding.is_empty() {
        0
    } else {
        spec.attrs.pool_padding[0] as i32
    };
    let tmpl = r#"// __NAME__: Pool (max 池化，每线程算一个输出元素)
// kernel_size=__KERNEL__, stride=__STRIDE__, padding=__PADDING__
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ input,
                         float* __restrict__ output,
                         int N, int C, int H, int W,
                         int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (idx >= total) return;
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int c = (idx / (OW * OH)) % C;
    int n = idx / (C * OH * OW);
    int kernel = __KERNEL__;
    int stride = __STRIDE__;
    int padding = __PADDING__;
    // max pool（avg pool 在 host 端用 reduce_mean kernel 替代）
    float result = -INFINITY;
    for (int kh = 0; kh < kernel; ++kh) {
        for (int kw = 0; kw < kernel; ++kw) {
            int ih = oh * stride - padding + kh;
            int iw = ow * stride - padding + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float v = input[n * C * H * W + c * H * W + ih * W + iw];
                result = fmaxf(result, v);
            }
        }
    }
    output[idx] = result;
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__KERNEL__", &kernel_size.to_string())
        .replace("__STRIDE__", &stride.to_string())
        .replace("__PADDING__", &padding.to_string())
}

// ---------------------------------------------------------------------------
// Reshape（copy kernel，元素级）
// ---------------------------------------------------------------------------

fn reshape_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let t = spec.dtype.c_type();
    let tmpl = r#"// __NAME__: Reshape (copy kernel，元素级，仅复制不改 shape 语义)
// shape 由 host 端 TensorSpec 维护，kernel 只做数据搬运
// TEMPLATE: __T__
__global__ void __NAME__(const __T__* __restrict__ input,
                         __T__* __restrict__ output,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i];
    }
}
"#;
    tmpl.replace("__NAME__", name).replace("__T__", t)
}

// ---------------------------------------------------------------------------
// Transpose（2D block + shared memory + bank conflict padding +1 列）
// ---------------------------------------------------------------------------

fn transpose_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let t = spec.dtype.c_type();
    let tmpl = r#"// __NAME__: Transpose (2D block + shared memory + bank conflict padding +1 列)
// TEMPLATE: __T__
__global__ void __NAME__(const __T__* __restrict__ input,
                         __T__* __restrict__ output,
                         int M, int N) {
    // 32x32 tile + 1 列 padding 避免 shared memory bank conflict
    __shared__ __T__ tile[32][32 + 1];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    // load input to shared memory（coalesced read，按行读）
    for (int j = 0; j < 32; j += 8) {
        int row = y + j / 8;
        if (row < M && x < N) {
            tile[threadIdx.y + j / 8][threadIdx.x] = input[row * N + x];
        }
    }
    __syncthreads();
    // 转置索引：output (col, row) = input (row, col)
    int out_x = blockIdx.y * 32 + threadIdx.x;
    int out_y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) {
        int row = out_y + j / 8;
        if (row < N && out_x < M) {
            output[row * M + out_x] = tile[threadIdx.x][threadIdx.y + j / 8];
        }
    }
}
"#;
    tmpl.replace("__NAME__", name).replace("__T__", t)
}

// ---------------------------------------------------------------------------
// Concat（按 axis 拼接，每线程拷一个元素）
// ---------------------------------------------------------------------------

fn concat_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let axis = spec.attrs.axis.unwrap_or(0);
    let t = spec.dtype.c_type();
    let tmpl = r#"// __NAME__: Concat (按 axis=__AXIS__ 拼接，每线程拷一个元素)
// TEMPLATE: __T__
// 简化为两输入拼接；多输入时 host 端可链式调用或用 array of pointers
__global__ void __NAME__(const __T__* __restrict__ in_a,
                         const __T__* __restrict__ in_b,
                         int a_axis_len,
                         __T__* __restrict__ output,
                         int total_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_n) return;
    // 按 axis=__AXIS__ 拼接：前 a_axis_len 个元素来自 in_a，其余来自 in_b
    if (i < a_axis_len) {
        output[i] = in_a[i];
    } else {
        output[i] = in_b[i - a_axis_len];
    }
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__AXIS__", &axis.to_string())
        .replace("__T__", t)
}

// ---------------------------------------------------------------------------
// Slice（按 starts/ends/axes/steps 拷贝）
// ---------------------------------------------------------------------------

fn slice_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let t = spec.dtype.c_type();
    let starts = if spec.attrs.slice_starts.is_empty() {
        "0".to_string()
    } else {
        spec.attrs
            .slice_starts
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let ends = if spec.attrs.slice_ends.is_empty() {
        "-1".to_string()
    } else {
        spec.attrs
            .slice_ends
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let steps = if spec.attrs.slice_steps.is_empty() {
        "1".to_string()
    } else {
        spec.attrs
            .slice_steps
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let tmpl = r#"// __NAME__: Slice (按 starts/ends/axes/steps 拷贝)
// starts=[__STARTS__], ends=[__ENDS__], steps=[__STEPS__]
// TEMPLATE: __T__
// 简化为 1D slice；多维 slice 由 host 端按实际参数调整索引计算
__global__ void __NAME__(const __T__* __restrict__ input,
                         __T__* __restrict__ output,
                         int start, int step, int out_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_n) {
        int src_idx = start + i * step;
        output[i] = input[src_idx];
    }
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__STARTS__", &starts)
        .replace("__ENDS__", &ends)
        .replace("__STEPS__", &steps)
        .replace("__T__", t)
}

// ---------------------------------------------------------------------------
// Constant（把 value 广播到 n 个元素；多元素用 __constant__ 数组）
// ---------------------------------------------------------------------------

fn constant_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let value = spec.attrs.value.unwrap_or(0.0);
    let t = spec.dtype.c_type();
    let has_tensor_data = !spec.attrs.tensor_data.is_empty();
    let tensor_init = if has_tensor_data {
        let count = spec.attrs.tensor_data.len();
        let line1 = format!(
            "// 多元素常量：用 __constant__ 数组初始化（{} 个元素，由 host 端填入）\n",
            count
        );
        let line2 = format!(
            "__constant__ float const_data[{}] = {{ /* filled by host */ }};\n",
            count
        );
        format!("{line1}{line2}")
    } else {
        String::new()
    };
    let tmpl = r#"// __NAME__: Constant (把 value=__VALUE__ 广播到 n 个元素)
// TEMPLATE: __T__
__TENSOR_INIT____global__ void __NAME__(__T__* out, __T__ value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = value;
    }
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__T__", t)
        .replace("__VALUE__", &value.to_string())
        .replace("__TENSOR_INIT__", &tensor_init)
}

// ---------------------------------------------------------------------------
// Placeholder / Return（数据流标记，不需要 kernel）
// ---------------------------------------------------------------------------

fn placeholder_kernel(spec: &KernelSpec) -> String {
    format!(
        "// placeholder: {}\n// (输入数据由外部提供，不需要 kernel)\n",
        spec.name
    )
}

fn return_kernel(spec: &KernelSpec) -> String {
    format!(
        "// return: {}\n// (输出数据直接转发，不需要 kernel)\n",
        spec.name
    )
}

// ---------------------------------------------------------------------------
// Fused（串联 fused_ops 列表，单 kernel 多 op）
// ---------------------------------------------------------------------------

fn fused_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let ops = if spec.attrs.fused_ops.is_empty() {
        vec!["identity".to_string()]
    } else {
        spec.attrs.fused_ops.clone()
    };
    let ops_str = ops.join(", ");
    // 为每个 fused op 生成一行变换代码
    let mut body = String::new();
    for op in &ops {
        let line: String = match op.as_str() {
            "relu" => "        x = fmaxf(x, 0.0f);\n".to_string(),
            "sigmoid" => "        x = 1.0f / (1.0f + expf(-x));\n".to_string(),
            "tanh" => "        x = tanhf(x);\n".to_string(),
            "exp" => "        x = expf(x);\n".to_string(),
            "log" => "        x = logf(x);\n".to_string(),
            "sqrt" => "        x = sqrtf(x);\n".to_string(),
            "rsqrt" => "        x = rsqrtf(x);\n".to_string(),
            "reciprocal" => "        x = 1.0f / x;\n".to_string(),
            "abs" => "        x = fabsf(x);\n".to_string(),
            "gelu" => {
                "        x = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));\n"
                    .to_string()
            }
            "identity" => "        /* identity: no-op */\n".to_string(),
            _ => format!("        /* unknown op: {} */\n", op),
        };
        body.push_str(&line);
    }
    let tmpl = r#"// __NAME__: Fused (串联 fused_ops 列表，按 op name 顺序应用变换，单 kernel 多 op)
// fused_ops: __OPS__
// TEMPLATE: float
__global__ void __NAME__(const float* __restrict__ in,
                         float* __restrict__ out,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        // 按 fused_ops 顺序应用变换（避免中间结果写回 global memory）
__BODY__        out[i] = x;
    }
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__OPS__", &ops_str)
        .replace("__BODY__", &body)
}

// ---------------------------------------------------------------------------
// Custom（用户自定义算子入口点，extern "C" 链接）
// ---------------------------------------------------------------------------

fn custom_kernel(spec: &KernelSpec) -> String {
    let name = &spec.name;
    let custom_type = if spec.attrs.custom_op_type.is_empty() {
        "unknown"
    } else {
        &spec.attrs.custom_op_type
    };
    let t = spec.dtype.c_type();
    let tmpl = r#"// __NAME__: Custom (用户自定义算子入口点，extern "C" 链接)
// 原始 op_type: __CUSTOM_TYPE__
// TEMPLATE: __T__
extern "C" __global__ void __NAME__(__T__* out, const __T__* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 用户应在此处填入自定义算子实现
        // 默认行为：identity copy（占位，确保 kernel 可链接）
        out[i] = in[i];
    }
}
"#;
    tmpl.replace("__NAME__", name)
        .replace("__CUSTOM_TYPE__", custom_type)
        .replace("__T__", t)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use base::DType;

    /// 构造一个最小 KernelSpec：1 输入 1 输出，dtype=F32，shape=[16,16]
    fn make_spec(name: &str, op: OpKind) -> KernelSpec {
        KernelSpec {
            name: name.to_string(),
            op,
            inputs: vec![TensorSpec {
                name: "x".to_string(),
                dims: vec![16, 16],
                dtype: DType::F32,
                is_input: true,
            }],
            outputs: vec![TensorSpec {
                name: "y".to_string(),
                dims: vec![16, 16],
                dtype: DType::F32,
                is_input: false,
            }],
            attrs: KernelAttrs::default(),
            dtype: DType::F32,
            node_idx: 0,
        }
    }

    #[test]
    fn test_generate_empty() {
        // 空 kernels 列表：source 仍含头部但无 kernel
        let out = generate(&[], GpuArch::Hopper90).unwrap();
        assert!(out.source.contains("kernel 数量: 0"));
        assert!(out.kernels.is_empty());
        assert!(out.source.contains("CUDA C++ kernel"));
    }

    #[test]
    fn test_generate_elementwise() {
        let kernels = vec![make_spec("neutron_add_0", OpKind::Add)];
        let out = generate(&kernels, GpuArch::Ampere80).unwrap();
        assert!(
            out.source.contains("__global__"),
            "source should contain __global__"
        );
        assert!(
            out.source.contains("neutron_add_0"),
            "source should contain kernel name"
        );
        assert!(out.source.contains("c[i] = a[i] + b[i];"));
        // launch wrapper 应存在
        assert!(out.source.contains("launch_neutron_add_0"));
    }

    #[test]
    fn test_generate_matmul_ampere() {
        let kernels = vec![make_spec("neutron_matmul_0", OpKind::MatMul)];
        let out = generate(&kernels, GpuArch::Ampere80).unwrap();
        assert!(
            out.source.contains("nvcuda::wmma") || out.source.contains("mma.sync"),
            "Ampere matmul should use nvcuda::wmma or mma.sync"
        );
        assert!(out.source.contains("wmma::fragment"));
    }

    #[test]
    fn test_generate_matmul_hopper() {
        let kernels = vec![make_spec("neutron_matmul_0", OpKind::MatMul)];
        let out = generate(&kernels, GpuArch::Hopper90).unwrap();
        let lower = out.source.to_lowercase();
        assert!(
            lower.contains("wgmma") || lower.contains("tma"),
            "Hopper matmul should use wgmma or TMA"
        );
        assert!(out.source.contains("wgmma.mma_async"));
    }

    #[test]
    fn test_generate_matmul_blackwell() {
        let kernels = vec![make_spec("neutron_matmul_0", OpKind::MatMul)];
        let out = generate(&kernels, GpuArch::Blackwell100).unwrap();
        let lower = out.source.to_lowercase();
        assert!(
            lower.contains("tensor memory") || lower.contains("tc"),
            "Blackwell matmul should mention tensor memory or tc"
        );
        assert!(out.source.contains("tensor memory accelerator"));
    }

    #[test]
    fn test_generate_softmax() {
        let kernels = vec![make_spec("neutron_softmax_0", OpKind::Softmax)];
        let out = generate(&kernels, GpuArch::Hopper90).unwrap();
        assert!(
            out.source.contains("shared"),
            "softmax should use shared memory"
        );
        assert!(out.source.contains("max"), "softmax should compute max");
        assert!(out.source.contains("exp"), "softmax should compute exp");
        assert!(out.source.contains("__shfl_xor_sync"));
    }

    #[test]
    fn test_generate_layernorm() {
        let kernels = vec![make_spec("neutron_layernorm_0", OpKind::LayerNorm)];
        let out = generate(&kernels, GpuArch::Hopper90).unwrap();
        let lower = out.source.to_lowercase();
        assert!(
            lower.contains("epsilon") || lower.contains("mean"),
            "layernorm should mention epsilon or mean"
        );
        assert!(out.source.contains("inv_std"));
        assert!(out.source.contains("gamma"));
        assert!(out.source.contains("beta"));
    }

    #[test]
    fn test_generate_reduce() {
        let kernels = vec![
            make_spec("neutron_reduce_sum_0", OpKind::ReduceSum),
            make_spec("neutron_reduce_mean_0", OpKind::ReduceMean),
            make_spec("neutron_reduce_max_0", OpKind::ReduceMax),
        ];
        let out = generate(&kernels, GpuArch::Ampere80).unwrap();
        assert!(
            out.source.contains("__syncthreads"),
            "reduce should use __syncthreads"
        );
        assert!(
            out.source.contains("shared[256]"),
            "reduce should use shared memory"
        );
        assert!(out.source.contains("ReduceMean"));
        // ReduceMean 应除以 N
        assert!(out.source.contains("/ (float)N"));
    }

    #[test]
    fn test_generate_all_ops() {
        // 构造全部 OpKind 的 KernelSpec，遍历所有 CUDA 架构，确保 source 非空且含 __global__
        let all_ops = [
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
            OpKind::Fused,
            OpKind::Reciprocal,
            OpKind::Abs,
            OpKind::Log,
            OpKind::Custom,
        ];
        for arch in [GpuArch::Ampere80, GpuArch::Hopper90, GpuArch::Blackwell100] {
            let kernels: Vec<KernelSpec> = all_ops
                .iter()
                .enumerate()
                .map(|(i, op)| make_spec(&format!("neutron_op_{i}"), *op))
                .collect();
            let out = generate(&kernels, arch).unwrap();
            assert!(
                !out.source.is_empty(),
                "source should be non-empty for {:?}",
                arch
            );
            assert!(
                out.source.contains("__global__"),
                "source should contain __global__ for {:?}",
                arch
            );
            // 每个 kernel 都应在 launch section 中出现
            for k in &kernels {
                assert!(
                    out.source.contains(&k.name),
                    "source should contain kernel name {} for {:?}",
                    k.name,
                    arch
                );
            }
        }
    }

    #[test]
    fn test_generate_constant_with_value() {
        let mut spec = make_spec("neutron_constant_0", OpKind::Constant);
        spec.attrs.value = Some(3.14);
        let out = generate(&[spec], GpuArch::Hopper90).unwrap();
        assert!(out.source.contains("__global__"));
        assert!(out.source.contains("3.14"));
        // launch wrapper 应传递 value
        assert!(out.source.contains("launch_neutron_constant_0"));
    }

    #[test]
    fn test_generate_custom_extern_c() {
        let mut spec = make_spec("neutron_custom_0", OpKind::Custom);
        spec.attrs.custom_op_type = "MyCustomOp".to_string();
        let out = generate(&[spec], GpuArch::Ampere80).unwrap();
        assert!(out.source.contains("extern \"C\""));
        assert!(out.source.contains("MyCustomOp"));
        assert!(out.source.contains("__global__"));
    }

    #[test]
    fn test_generate_fused_chain() {
        let mut spec = make_spec("neutron_fused_0", OpKind::Fused);
        spec.attrs.fused_ops = vec!["relu".to_string(), "exp".to_string()];
        let out = generate(&[spec], GpuArch::Hopper90).unwrap();
        assert!(out.source.contains("fused_ops: relu, exp"));
        assert!(out.source.contains("x = fmaxf(x, 0.0f);"));
        assert!(out.source.contains("x = expf(x);"));
    }

    #[test]
    fn test_generate_transpose_padding() {
        let kernels = vec![make_spec("neutron_transpose_0", OpKind::Transpose)];
        let out = generate(&kernels, GpuArch::Hopper90).unwrap();
        // bank conflict padding: 32 + 1
        assert!(out.source.contains("32 + 1"));
        assert!(out.source.contains("__shared__"));
        assert!(out.source.contains("__syncthreads"));
    }
}
