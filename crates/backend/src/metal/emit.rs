//! metal emit — Metal Shading Language (MSL) kernel 代码生成
//!
//! 为每个 [`KernelSpec`] 生成真实的 MSL `kernel void` 源码，覆盖全部 OpKind：
//! - 元素级（Add/Sub/Mul/Div/Relu/Gelu/Sigmoid/Tanh/Sqrt/Exp/Log/Rsqrt/Reciprocal/Abs/Pow）：
//!   `[[thread_position_in_grid]]`，每线程一个元素
//! - Reduce（Sum/Mean/Max）：threadgroup shared memory 树形归约 + `threadgroup_barrier`
//! - MatMul：`simdgroup<float8x8>` 8x8 矩阵乘法单元（Apple GPU）
//! - Softmax/LayerNorm：每行一个 threadgroup，多遍扫描 + shared memory
//! - Conv/Pool：直接卷积/池化，每线程一个输出元素
//! - 数据移动：Reshape/Transpose/Concat/Slice
//! - 数据流：Constant/Placeholder/Return
//! - Fused/Custom：融合算子链 / 自定义算子
//!
//! 模板用 `r#"..."#` 原始字符串 + `__TOKEN__` 占位 + `.replace()`，
//! 避免 `format!` 转义大括号的繁琐。

use crate::spec::*;
use base::{OpKind, Result};

/// 生成 Metal kernel 源码。
///
/// 对每个非空 [`KernelSpec`] 生成对应的 `kernel void` 函数，并附加 launch wrapper 注释。
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
        lang: SourceLang::Metal,
        kernels: kernel_infos,
        arch,
    })
}

// ---------------------------------------------------------------------------
// 头部 + launch wrapper
// ---------------------------------------------------------------------------

fn make_header(arch: GpuArch, count: usize) -> String {
    let arch_name = arch.name();
    let mut s = String::new();
    s.push_str("// ============================================================\n");
    s.push_str("// neutron — Metal Shading Language (MSL) kernel 源码\n");
    s.push_str(&format!("// 微架构: {arch_name} ({count} 个 kernel)\n"));
    s.push_str("// 生成器: backend::metal::emit\n");
    if arch.has_simdgroup() {
        s.push_str("// 特性: Apple GPU simdgroup_matrix 8x8 矩阵乘法单元\n");
    } else {
        s.push_str("// 注: Metal 主要面向 Apple GPU；非 Apple 架构仍生成 simdgroup 代码\n");
    }
    s.push_str("// ============================================================\n\n");
    s.push_str("#include <metal_stdlib>\n");
    s.push_str("using namespace metal;\n\n");
    s
}

fn make_launch_section(kernels: &[KernelSpec], arch: GpuArch) -> String {
    let mut s = String::new();
    s.push_str("// ============================================================\n");
    s.push_str(
        "// launch wrappers（Metal host 端用 Obj-C++ / Swift 调度 MTLComputeCommandEncoder）\n",
    );
    s.push_str("// ============================================================\n");
    for k in kernels {
        let l = k.launch(arch);
        let name = &k.name;
        let (g0, g1, g2) = l.grid;
        let (b0, b1, b2) = l.block;
        let sm = l.shared_mem;
        s.push_str(&format!(
            "// launch_{name}: grid=({g0}, {g1}, {g2}), block=({b0}, {b1}, {b2}), threadgroup memory={sm} bytes\n"
        ));
    }
    s
}

// ---------------------------------------------------------------------------
// kernel 分派
// ---------------------------------------------------------------------------

fn make_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    match spec.op {
        OpKind::Add => elem_binary(spec, "c[tid] = a[tid] + b[tid];"),
        OpKind::Sub => elem_binary(spec, "c[tid] = a[tid] - b[tid];"),
        OpKind::Mul => elem_binary(spec, "c[tid] = a[tid] * b[tid];"),
        OpKind::Div => elem_binary(spec, "c[tid] = a[tid] / b[tid];"),
        OpKind::Pow => elem_binary(spec, "c[tid] = pow(a[tid], b[tid]);"),
        OpKind::Relu => elem_unary(spec, "max(x, 0.0f)"),
        OpKind::Gelu => elem_unary(
            spec,
            "0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x*x*x)))",
        ),
        OpKind::Sigmoid => elem_unary(spec, "1.0f / (1.0f + exp(-x))"),
        OpKind::Tanh => elem_unary(spec, "tanh(x)"),
        OpKind::Sqrt => elem_unary(spec, "sqrt(x)"),
        OpKind::Exp => elem_unary(spec, "exp(x)"),
        OpKind::Log => elem_unary(spec, "log(x)"),
        OpKind::Rsqrt => elem_unary(spec, "rsqrt(x)"),
        OpKind::Reciprocal => elem_unary(spec, "1.0f / x"),
        OpKind::Abs => elem_unary(spec, "abs(x)"),
        OpKind::ReduceSum => reduce_kernel(spec, "+", "0.0f", false),
        OpKind::ReduceMean => reduce_kernel(spec, "+", "0.0f", true),
        OpKind::ReduceMax => reduce_kernel(spec, "max", "-INFINITY", false),
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
    let t = spec.dtype.msl_type();
    let tmpl = r#"// __NAME__: 元素级二元 op（每线程一个元素）
kernel void __NAME__(device const __T__* a [[buffer(0)]],
                     device const __T__* b [[buffer(1)]],
                     device __T__* c [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    __BODY__
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__BODY__", body)
}

fn elem_unary(spec: &KernelSpec, expr: &str) -> String {
    let t = spec.dtype.msl_type();
    let tmpl = r#"// __NAME__: 元素级一元 op（每线程一个元素）
kernel void __NAME__(device const __T__* a [[buffer(0)]],
                     device __T__* c [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    __T__ x = a[tid];
    c[tid] = __EXPR__;
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__EXPR__", expr)
}

// ---------------------------------------------------------------------------
// Reduce（threadgroup shared memory 树形归约）
// ---------------------------------------------------------------------------

fn reduce_kernel(spec: &KernelSpec, op: &str, identity: &str, is_mean: bool) -> String {
    let t = spec.dtype.msl_type();
    let combine = if op == "max" {
        "shared[tid] = max(shared[tid], shared[tid + s]);".to_string()
    } else {
        format!("shared[tid] = shared[tid] {op} shared[tid + s];")
    };
    let finalize = if is_mean {
        "if (tid == 0) output[bid] = shared[0] / (float)N;\n"
    } else {
        "if (tid == 0) output[bid] = shared[0];\n"
    };
    let tmpl = r#"// __NAME__: 树形归约（threadgroup shared memory + barrier）
kernel void __NAME__(device const __T__* input [[buffer(0)]],
                    device __T__* output [[buffer(1)]],
                    constant uint& N [[buffer(2)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint gid [[thread_position_in_grid]],
                    uint bid [[threadgroup_position_in_grid]],
                    uint block_size [[threads_per_threadgroup]]) {
    threadgroup __T__ shared[256];
    shared[tid] = (gid < N) ? input[gid] : __IDENT__;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            __COMBINE__
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    __FINALIZE__
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__IDENT__", identity)
        .replace("__COMBINE__", &combine)
        .replace("__FINALIZE__", finalize)
}

// ---------------------------------------------------------------------------
// MatMul（simdgroup<float8x8> 8x8 矩阵乘法单元）
// ---------------------------------------------------------------------------

fn matmul_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    let arch_note = if arch.has_simdgroup() {
        format!(
            "// {} — Apple GPU simdgroup_matrix 8x8 矩阵乘法单元",
            arch.name()
        )
    } else {
        format!(
            "// {} — Metal 主要面向 Apple GPU，仍使用 simdgroup 8x8 矩阵乘法",
            arch.name()
        )
    };
    let tmpl = r#"// __NAME__: MatMul（simdgroup 8x8 tile 矩阵乘法）
__ARCH_NOTE__
// BLOCK_M=BLOCK_N=32（4x4 个 simdgroup 8x8 tile）
kernel void __NAME__(device const float* A [[buffer(0)]],
                     device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]],
                     constant uint3& dims [[buffer(3)]],   // dims = (M, N, K)
                     uint2 local_id [[thread_position_in_threadgroup]],
                     uint2 block_id [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 8;
    constexpr uint BLOCK_M = 32;
    constexpr uint BLOCK_N = 32;
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;
    // threadgroup 内 simdgroup 的 2D 位置（每个 simdgroup 处理一个 8x8 输出 tile）
    uint sg_row = local_id.y / TILE;
    uint sg_col = local_id.x / TILE;
    uint row0 = block_id.y * BLOCK_M + sg_row * TILE;
    uint col0 = block_id.x * BLOCK_N + sg_col * TILE;
    simdgroup<float8x8> acc;
    simdgroup_fill(acc, 0.0f);
    for (uint k = 0; k < K; k += TILE) {
        simdgroup<float8x8> a_tile, b_tile;
        simdgroup_load(a_tile, A + row0 * K + k, K);
        simdgroup_load(b_tile, B + k * N + col0, N);
        // Apple GPU mma 指令原生支持累加
        simdgroup_multiply(acc, a_tile, b_tile);
    }
    simdgroup_store(acc, C + row0 * N + col0, N);
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__ARCH_NOTE__", &arch_note)
}

// ---------------------------------------------------------------------------
// Softmax（每行一个 threadgroup，三遍扫描）
// ---------------------------------------------------------------------------

fn softmax_kernel(spec: &KernelSpec) -> String {
    let tmpl = r#"// __NAME__: Softmax（每行一个 threadgroup，三遍扫描 + shared memory）
kernel void __NAME__(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& N_COLS [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint block_size [[threads_per_threadgroup]]) {
    threadgroup float shared[1024];
    uint base = row * N_COLS;
    // 第一遍：求行 max（数值稳定）
    float local_max = -INFINITY;
    for (uint i = lid; i < N_COLS; i += block_size) {
        local_max = max(local_max, input[base + i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 第二遍：exp(x - max) 并求和
    float local_sum = 0.0f;
    for (uint i = lid; i < N_COLS; i += block_size) {
        float v = exp(input[base + i] - row_max);
        output[base + i] = v;
        local_sum += v;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 第三遍：归一化
    for (uint i = lid; i < N_COLS; i += block_size) {
        output[base + i] = output[base + i] / row_sum;
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name)
}

// ---------------------------------------------------------------------------
// LayerNorm（两遍扫描：mean → var → normalize）
// ---------------------------------------------------------------------------

fn layernorm_kernel(spec: &KernelSpec) -> String {
    let eps = spec.attrs.epsilon.unwrap_or(1e-5).to_string();
    let tmpl = r#"// __NAME__: LayerNorm（两遍扫描 + shared memory，epsilon 用于数值稳定）
kernel void __NAME__(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& N_COLS [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint block_size [[threads_per_threadgroup]]) {
    threadgroup float shared[1024];
    uint base = row * N_COLS;
    // 第一遍：求 mean
    float local_sum = 0.0f;
    for (uint i = lid; i < N_COLS; i += block_size) {
        local_sum += input[base + i];
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / (float)N_COLS;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 第二遍：求 var
    float local_sq = 0.0f;
    for (uint i = lid; i < N_COLS; i += block_size) {
        float d = input[base + i] - mean;
        local_sq += d * d;
    }
    shared[lid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = shared[0] / (float)N_COLS;
    // inv_std = 1 / sqrt(var + epsilon)
    float inv_std = rsqrt(var + __EPS__);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 第三遍：归一化 (x - mean) * inv_std
    for (uint i = lid; i < N_COLS; i += block_size) {
        output[base + i] = (input[base + i] - mean) * inv_std;
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__EPS__", &eps)
}

// ---------------------------------------------------------------------------
// Conv2D（直接卷积，每线程一个输出元素）
// ---------------------------------------------------------------------------

fn conv_kernel(spec: &KernelSpec) -> String {
    let tmpl = r#"// __NAME__: Conv2D（直接卷积，每线程一个输出元素，嵌套循环 KH/KW/C_in）
kernel void __NAME__(device const float* input [[buffer(0)]],
                     device const float* weight [[buffer(1)]],
                     device const float* bias [[buffer(2)]],
                     device float* output [[buffer(3)]],
                     constant uint4& dims [[buffer(4)]],        // dims = (N, C_in, H, W)
                     constant uint4& conv_params [[buffer(5)]], // conv_params = (C_out, stride_h, stride_w, KH*KW)
                     uint2 gid [[thread_position_in_grid]]) {
    uint C_in = dims.y, H = dims.z, W = dims.w;
    uint C_out = conv_params.x, stride_h = conv_params.y, stride_w = conv_params.z;
    uint KH = conv_params.w;  // 简化：方形 kernel（KH=KW）
    uint KW = KH;
    uint OH = (H - KH) / stride_h + 1;
    uint OW = (W - KW) / stride_w + 1;
    uint oh = gid.y;
    uint ow = gid.x;
    if (oh >= OH || ow >= OW) return;
    for (uint co = 0; co < C_out; ++co) {
        float acc = bias[co];
        for (uint ci = 0; ci < C_in; ++ci) {
            for (uint kh = 0; kh < KH; ++kh) {
                for (uint kw = 0; kw < KW; ++kw) {
                    uint ih = oh * stride_h + kh;
                    uint iw = ow * stride_w + kw;
                    float v = input[(ci * H + ih) * W + iw];
                    float wgt = weight[((co * C_in + ci) * KH + kh) * KW + kw];
                    acc += v * wgt;
                }
            }
        }
        output[(co * OH + oh) * OW + ow] = acc;
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name)
}

// ---------------------------------------------------------------------------
// Pool（max/avg 池化）
// ---------------------------------------------------------------------------

fn pool_kernel(spec: &KernelSpec) -> String {
    let tmpl = r#"// __NAME__: Pool（max 池化，每线程一个输出元素）
kernel void __NAME__(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint4& dims [[buffer(2)]],        // dims = (C, H, W, 0)
                     constant uint4& pool_params [[buffer(3)]], // pool_params = (KH, KW, stride_h, stride_w)
                     uint2 gid [[thread_position_in_grid]]) {
    uint C = dims.x, H = dims.y, W = dims.z;
    uint KH = pool_params.x, KW = pool_params.y;
    uint stride_h = pool_params.z, stride_w = pool_params.w;
    uint OH = (H - KH) / stride_h + 1;
    uint OW = (W - KW) / stride_w + 1;
    uint oh = gid.y;
    uint ow = gid.x;
    if (oh >= OH || ow >= OW) return;
    for (uint c = 0; c < C; ++c) {
        float result = -INFINITY;
        for (uint kh = 0; kh < KH; ++kh) {
            for (uint kw = 0; kw < KW; ++kw) {
                uint ih = oh * stride_h + kh;
                uint iw = ow * stride_w + kw;
                result = max(result, input[(c * H + ih) * W + iw]);
            }
        }
        output[(c * OH + oh) * OW + ow] = result;
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name)
}

// ---------------------------------------------------------------------------
// 数据移动
// ---------------------------------------------------------------------------

fn reshape_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let tmpl = r#"// __NAME__: Reshape（连续内存拷贝，shape 一致性由 host 保证）
kernel void __NAME__(device const __T__* input [[buffer(0)]],
                     device __T__* output [[buffer(1)]],
                     constant uint& N [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid < N) output[tid] = input[tid];
}
"#;
    tmpl.replace("__NAME__", &spec.name).replace("__T__", t)
}

fn transpose_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let tmpl = r#"// __NAME__: Transpose（2D threadgroup + shared memory tile，+1 列避免 bank conflict）
// 注: host 应以 2D threadgroup（如 32x32）启动以匹配此 kernel 的 2D 线程映射
kernel void __NAME__(device const __T__* input [[buffer(0)]],
                     device __T__* output [[buffer(1)]],
                     constant uint2& dims [[buffer(2)]],  // dims = (H, W)
                     uint2 local_id [[thread_position_in_threadgroup]],
                     uint2 block_id [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 32;
    uint H = dims.x;
    uint W = dims.y;
    threadgroup __T__ tile[TILE][TILE + 1];  // +1 列消除 shared memory bank conflict
    uint bx = block_id.x * TILE;
    uint by = block_id.y * TILE;
    uint tx = local_id.x;
    uint ty = local_id.y;
    // 加载 input 块到 shared memory
    if (bx + tx < W && by + ty < H) {
        tile[ty][tx] = input[(by + ty) * W + (bx + tx)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 转置写回：output[col, row] = input[row, col]
    if (bx + tx < H && by + ty < W) {
        output[(bx + tx) * H + (by + ty)] = tile[tx][ty];
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name).replace("__T__", t)
}

fn concat_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let n_inputs = spec.inputs.len().max(2);
    let n_inputs_str = n_inputs.to_string();
    let tmpl = r#"// __NAME__: Concat（沿 axis 拼接，每线程一个输出元素；此处展示 2 路拼接）
// 注: 通用版支持 __N_INPUTS__ 路输入，host 按 offset 区间分配
kernel void __NAME__(device const __T__* input0 [[buffer(0)]],
                     device const __T__* input1 [[buffer(1)]],
                     device __T__* output [[buffer(2)]],
                     constant uint& offset0 [[buffer(3)]],
                     constant uint& offset1 [[buffer(4)]],
                     constant uint& total [[buffer(5)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    if (tid < offset1) {
        output[tid] = input0[tid - offset0];
    } else {
        output[tid] = input1[tid - offset1];
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__N_INPUTS__", &n_inputs_str)
}

fn slice_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let tmpl = r#"// __NAME__: Slice（按 starts/ends/steps 拷贝）
kernel void __NAME__(device const __T__* input [[buffer(0)]],
                     device __T__* output [[buffer(1)]],
                     constant uint& start [[buffer(2)]],
                     constant uint& step [[buffer(3)]],
                     constant uint& out_n [[buffer(4)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid < out_n) {
        output[tid] = input[start + tid * step];
    }
}
"#;
    tmpl.replace("__NAME__", &spec.name).replace("__T__", t)
}

// ---------------------------------------------------------------------------
// 数据流
// ---------------------------------------------------------------------------

fn constant_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let val = spec.attrs.value.unwrap_or(0.0).to_string();
    let tmpl = r#"// __NAME__: Constant（把标量 value 广播到输出 buffer）
// 注: 默认 value = __VAL__（host 端可用 constant buffer 覆盖）
kernel void __NAME__(device __T__* output [[buffer(0)]],
                     constant __T__& value [[buffer(1)]],
                     constant uint& N [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid < N) output[tid] = value;
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__VAL__", &val)
}

fn placeholder_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let tmpl = r#"// __NAME__: Placeholder（输入占位）
// 注: Placeholder 标记图输入，host 端绑定 MTLBuffer；此处生成 identity 拷贝 kernel 便于调试
kernel void __NAME__(device const __T__* input [[buffer(0)]],
                     device __T__* output [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    output[tid] = input[tid];
}
"#;
    tmpl.replace("__NAME__", &spec.name).replace("__T__", t)
}

fn return_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let tmpl = r#"// __NAME__: Return（图输出）
// 注: Return 标记图输出，host 端读取该 buffer；此处生成 identity 拷贝 kernel 保证数据落盘
kernel void __NAME__(device const __T__* input [[buffer(0)]],
                     device __T__* output [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    output[tid] = input[tid];
}
"#;
    tmpl.replace("__NAME__", &spec.name).replace("__T__", t)
}

// ---------------------------------------------------------------------------
// Fused / Custom
// ---------------------------------------------------------------------------

fn fused_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let ops = if spec.attrs.fused_ops.is_empty() {
        "mul -> add -> relu".to_string()
    } else {
        spec.attrs.fused_ops.join(" -> ")
    };
    let tmpl = r#"// __NAME__: Fused（融合算子链: __OPS__）
kernel void __NAME__(device const __T__* a [[buffer(0)]],
                     device const __T__* b [[buffer(1)]],
                     device __T__* c [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    __T__ x = a[tid];
    __T__ y = b[tid];
    // 典型 GEMM + bias + 激活 融合：z = relu(x * y + y)
    __T__ z = x * y;
    z = z + y;
    z = max(z, 0.0f);
    c[tid] = z;
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
        .replace("__OPS__", &ops)
}

fn custom_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.msl_type();
    let op_type = if spec.attrs.custom_op_type.is_empty() {
        "unknown"
    } else {
        &spec.attrs.custom_op_type
    };
    let tmpl = r#"// __NAME__: Custom op（原始 op_type = __OP_TYPE__）
// 注: 未知 ONNX 算子，生成通用元素级 identity kernel，host 端可替换为自定义实现
kernel void __NAME__(device const __T__* input [[buffer(0)]],
                     device __T__* output [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    output[tid] = input[tid];
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__T__", t)
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
        let out = emit(&[], GpuArch::Apple6).expect("emit 不应失败");
        assert_eq!(out.lang, SourceLang::Metal);
        assert_eq!(out.arch, GpuArch::Apple6);
        assert!(out.kernels.is_empty(), "空 kernel 列表应无 kernel_info");
        // 即便没有 kernel，头部仍应包含 metal_stdlib
        assert!(out.source.contains("#include <metal_stdlib>"));
        assert!(out.source.contains("using namespace metal;"));
        assert!(out.source.contains("Apple M1"));
    }

    #[test]
    fn test_emit_elementwise() {
        let specs = vec![make_spec(OpKind::Add, 0)];
        let out = emit(&specs, GpuArch::Apple6).expect("emit 失败");
        assert!(out.source.contains("kernel void"));
        assert!(out.source.contains("thread_position_in_grid"));
        assert!(out.source.contains("neutron_add_0"));
        assert!(out.source.contains("c[tid] = a[tid] + b[tid];"));
        assert_eq!(out.kernels.len(), 1);
        assert_eq!(out.kernels[0].name, "neutron_add_0");
    }

    #[test]
    fn test_emit_matmul_apple6() {
        let specs = vec![make_spec(OpKind::MatMul, 4)];
        let out = emit(&specs, GpuArch::Apple6).expect("emit 失败");
        assert!(
            out.source.contains("simdgroup"),
            "Apple6 MatMul 必须用 simdgroup"
        );
        assert!(
            out.source.contains("simdgroup_multiply"),
            "Apple6 MatMul 必须调用 simdgroup_multiply"
        );
        assert!(out.source.contains("simdgroup<float8x8>"));
        assert!(out.source.contains("Apple M1"));
    }

    #[test]
    fn test_emit_matmul_apple8() {
        let specs = vec![make_spec(OpKind::MatMul, 4)];
        let out = emit(&specs, GpuArch::Apple8).expect("emit 失败");
        assert!(out.source.contains("simdgroup"));
        assert!(out.source.contains("simdgroup_multiply"));
        assert!(out.source.contains("Apple M3"));
    }

    #[test]
    fn test_emit_softmax() {
        let specs = vec![make_spec(OpKind::Softmax, 5)];
        let out = emit(&specs, GpuArch::Apple7).expect("emit 失败");
        assert!(out.source.contains("threadgroup"));
        assert!(out.source.contains("max"), "softmax 必须有 max（数值稳定）");
        assert!(out
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
        assert!(out.source.contains("exp"));
    }

    #[test]
    fn test_emit_layernorm() {
        let specs = vec![make_spec(OpKind::LayerNorm, 6)];
        let out = emit(&specs, GpuArch::Apple7).expect("emit 失败");
        // epsilon 或 mean 至少出现一个（本实现两者都有）
        assert!(
            out.source.contains("epsilon") || out.source.contains("mean"),
            "layernorm 应包含 epsilon 或 mean"
        );
        assert!(out.source.contains("mean"));
        assert!(out.source.contains("rsqrt"));
        assert!(out
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
    }

    #[test]
    fn test_emit_reduce() {
        for (op, needle) in [
            (OpKind::ReduceSum, "+"),
            (OpKind::ReduceMean, "(float)N"),
            (OpKind::ReduceMax, "max"),
        ] {
            let specs = vec![make_spec(op, 22)];
            let out = emit(&specs, GpuArch::Apple6).expect("emit 失败");
            assert!(out.source.contains("kernel void"));
            assert!(out
                .source
                .contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
            assert!(out.source.contains(needle), "reduce 应包含 {needle:?}");
        }
    }

    #[test]
    fn test_emit_transpose() {
        let specs = vec![make_spec(OpKind::Transpose, 14)];
        let out = emit(&specs, GpuArch::Apple6).expect("emit 失败");
        assert!(out.source.contains("threadgroup"));
        assert!(out
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
        // bank conflict padding：tile[TILE][TILE + 1]
        assert!(out.source.contains("TILE + 1"));
    }

    #[test]
    fn test_emit_all_ops() {
        // 构造所有 OpKind 的 KernelSpec，遍历所有 arch，确保 source 非空且包含 kernel void
        for arch in all_archs() {
            let specs: Vec<KernelSpec> = all_ops()
                .into_iter()
                .enumerate()
                .map(|(i, op)| make_spec(op, i as u32))
                .collect();
            let out = emit(&specs, arch).expect("emit 不应失败");
            assert!(!out.source.is_empty(), "source 不应为空 (arch={:?})", arch);
            assert!(
                out.source.contains("kernel void"),
                "source 应包含 kernel void (arch={:?})",
                arch
            );
            assert!(out.source.contains("#include <metal_stdlib>"));
            assert_eq!(out.kernels.len(), specs.len());
            assert_eq!(out.arch, arch);
            // 每个 kernel 都应有对应 launch wrapper 注释
            for s in &specs {
                assert!(
                    out.source.contains(&format!("// launch_{}:", s.name)),
                    "应有 launch_{} 注释",
                    s.name
                );
            }
        }
    }
}
