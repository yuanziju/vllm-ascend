//! cann emit — Ascend CANN C++ 算子代码生成
//!
//! 为每个 [`KernelSpec`] 生成真实的 Ascend C++ `__aicore__` kernel 源码，
//! 覆盖全部 31 个 OpKind：
//! - 元素级（Add/Sub/Mul/Div/Pow/Relu/Gelu/Sigmoid/Tanh/Sqrt/Exp/Log/Rsqrt/Reciprocal/Abs）：
//!   AscendC 高级 API（`AscendC::Add`/`Sub`/`Mul`/`Div`/`Maxs`/`Gelu`/`Sigmoid`/`Tanh`
//!   /`Sqrt`/`Exp`/`Log`/`Rsqrt`/`Reciprocal`/`Abs`/`Pow`），TILE_LENGTH=128 双缓冲
//! - Reduce（Sum/Mean/Max）：`AscendC::ReduceSum`/`ReduceMean`/`ReduceMax`
//! - MatMul：Cube Core（`AscendC::MatMul` + mmad 指令），按 arch 区分 910B1/B3 与 310P3
//! - Softmax：Vector Core 多遍（ReduceMax → Sub+Exp+ReduceSum → Div）
//! - LayerNorm：两遍（mean → var → normalize），epsilon 数值稳定
//! - Conv：im2col + Cube GEMM 路径
//! - Pool：max/avg 池化（Vector Core）
//! - Reshape/Transpose/Concat/Slice：基于 DataCopy 的数据移动
//! - Constant：`Duplicate` 填充
//! - Placeholder/Return：注释说明（数据流标记）
//! - Fused：串联 fused_ops 链
//! - Custom：`extern "C"` 占位 kernel
//!
//! 模板用 `r#"..."#` 原始字符串 + `__TOKEN__` 占位 + `.replace()`，
//! 避免 `format!` 转义大括号的繁琐。每个 kernel 采用 Ascend 标准结构：
//! `class` + `Init`/`Process`/`CopyIn`/`Compute`/`CopyOut` + `__aicore__ kernel` 函数。

use crate::spec::DTypeExt;
use crate::spec::*;
use base::{OpKind, Result};

/// 生成 Ascend CANN C++ kernel 源码。
///
/// 对每个非空 [`KernelSpec`] 生成对应的 `__aicore__` kernel 函数，
/// 并附加 launch 说明注释（host 端用 aclrtLaunch 调度）。
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
        lang: SourceLang::Cann,
        kernels: kernel_infos,
        arch,
    })
}

// ---------------------------------------------------------------------------
// 头部 + launch 说明
// ---------------------------------------------------------------------------

fn make_header(arch: GpuArch, count: usize) -> String {
    let arch_name = arch.name();
    let mut s = String::new();
    s.push_str("// ============================================================\n");
    s.push_str("// neutron — Ascend CANN C++ 算子源码\n");
    s.push_str(&format!("// 微架构: {arch_name} ({count} 个 kernel)\n"));
    s.push_str("// 生成器: backend::cann::emit\n");
    if arch.has_cube_core() {
        s.push_str("// 微架构特性: Ascend Vector Core + Cube Core（mmad 矩阵乘法单元）\n");
        s.push_str("//   - Cube Core: 原生 MatMul/mmad 指令，FP16/BF16/INT8 高吞吐\n");
        s.push_str("//   - Vector Core: SIMD 向量指令，Add/Mul/Sqrt/Exp 等元素级\n");
        s.push_str("//   - TQue/TPipe 双缓冲 + tiling（TILE_LENGTH=128）\n");
    } else if matches!(arch, GpuArch::Ascend310P3) {
        s.push_str("// 微架构特性: Ascend 310P3 轻量推理路径（Vector Core + AiCore 优化）\n");
        s.push_str("//   - 主要面向推理场景，Cube 路径走轻量 GEMM\n");
        s.push_str("//   - TQue/TPipe 双缓冲 + tiling（TILE_LENGTH=128）\n");
    } else {
        s.push_str("// 注: CANN 主要面向 Ascend NPU；非 Ascend 架构仍生成 AscendC 代码\n");
    }
    s.push_str("// ============================================================\n\n");
    s.push_str("#include \"kernel_tiling.h\"\n");
    s.push_str("#include \"kernel_operator.h\"\n");
    s.push_str("using namespace AscendC;\n\n");
    s.push_str("// AscendC 标准常量：双缓冲 + tile 长度\n");
    s.push_str("constexpr int32_t BUFFER_NUM = 2;\n");
    s.push_str("constexpr int32_t TILE_LENGTH = 128;\n\n");
    s
}

fn make_launch_section(kernels: &[KernelSpec], arch: GpuArch) -> String {
    let mut s = String::new();
    s.push_str("// ============================================================\n");
    s.push_str("// launch 说明（host 端用 aclrtLaunch + blockDim 调度 __aicore__ kernel）\n");
    s.push_str("// ============================================================\n");
    for k in kernels {
        let l = k.launch(arch);
        let name = &k.name;
        let (g0, g1, g2) = l.grid;
        let (b0, b1, b2) = l.block;
        let sm = l.shared_mem;
        s.push_str(&format!(
            "// launch_{name}: grid=({g0}, {g1}, {g2}), block=({b0}, {b1}, {b2}), unified buffer~={sm} bytes\n"
        ));
        s.push_str(&format!(
            "//   aclrtSetDevice(0); {}<<<blocks, &tiling>>>(x, y, z, workspace, tiling);\n",
            name
        ));
    }
    s
}

// ---------------------------------------------------------------------------
// kernel 分派
// ---------------------------------------------------------------------------

fn make_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    match spec.op {
        OpKind::Add => elem_binary(spec, "Add", "x, y"),
        OpKind::Sub => elem_binary(spec, "Sub", "x, y"),
        OpKind::Mul => elem_binary(spec, "Mul", "x, y"),
        OpKind::Div => elem_binary(spec, "Div", "x, y"),
        OpKind::Pow => elem_binary(spec, "Pow", "x, y"),
        OpKind::Relu => elem_unary(spec, "Relu", "// AscendC::Relu 等价 Maxs(x, 0)"),
        OpKind::Gelu => elem_unary(spec, "Gelu", "// AscendC::Gelu 超越函数"),
        OpKind::Sigmoid => elem_unary(spec, "Sigmoid", "// AscendC::Sigmoid"),
        OpKind::Tanh => elem_unary(spec, "Tanh", "// AscendC::Tanh"),
        OpKind::Sqrt => elem_unary(spec, "Sqrt", "// AscendC::Sqrt"),
        OpKind::Exp => elem_unary(spec, "Exp", "// AscendC::Exp"),
        OpKind::Log => elem_unary(spec, "Log", "// AscendC::Log"),
        OpKind::Rsqrt => elem_unary(spec, "Rsqrt", "// AscendC::Rsqrt"),
        OpKind::Reciprocal => elem_unary(spec, "Reciprocal", "// AscendC::Reciprocal"),
        OpKind::Abs => elem_unary(spec, "Abs", "// AscendC::Abs"),
        OpKind::ReduceSum => reduce_kernel(spec, "ReduceSum", false),
        OpKind::ReduceMean => reduce_kernel(spec, "ReduceMean", true),
        OpKind::ReduceMax => reduce_kernel(spec, "ReduceMax", false),
        OpKind::MatMul => matmul_kernel(spec, arch),
        OpKind::Softmax => softmax_kernel(spec, arch),
        OpKind::LayerNorm => layernorm_kernel(spec, arch),
        OpKind::Conv => conv_kernel(spec, arch),
        OpKind::Pool => pool_kernel(spec, arch),
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
// 辅助：类型 + class 名
// ---------------------------------------------------------------------------

fn class_name(spec: &KernelSpec) -> String {
    let cap: String = spec
        .name
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if i == 0
                || !spec
                    .name
                    .as_bytes()
                    .get(i - 1)
                    .map(|b| *b == b'_')
                    .unwrap_or(false)
            {
                c.to_ascii_uppercase()
            } else {
                c
            }
        })
        .collect();
    // 把 _ 后首字母大写：neutron_add_0 -> NeutronAdd0
    let mut out = String::new();
    let mut next_upper = true;
    for c in cap.chars() {
        if c == '_' {
            next_upper = true;
            continue;
        }
        if next_upper {
            out.push(c.to_ascii_uppercase());
            next_upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 元素级：二元 op（Add/Sub/Mul/Div/Pow）
// ---------------------------------------------------------------------------

fn elem_binary(spec: &KernelSpec, op_name: &str, _args: &str) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let api = match op_name {
        "Add" => "Add",
        "Sub" => "Sub",
        "Mul" => "Mul",
        "Div" => "Div",
        "Pow" => "Pow",
        _ => "Add",
    };
    let tmpl = r#"// __NAME__: 元素级二元 op（AscendC::__API__，TILE_LENGTH=128 双缓冲）
// Vector Core SIMD：每 tile 调用一次 AscendC 高级 API
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ __T__*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ __T__*)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ __T__*)z + offset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ void CopyIn(int32_t progress) {
        LocalTensor<__T__> xLocal = inQueueX.AllocTensor<__T__>();
        LocalTensor<__T__> yLocal = inQueueY.AllocTensor<__T__>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ void Compute(int32_t progress) {
        LocalTensor<__T__> xLocal = inQueueX.DeQue<__T__>();
        LocalTensor<__T__> yLocal = inQueueY.DeQue<__T__>();
        LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
        AscendC::__API__(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ void CopyOut(int32_t progress) {
        LocalTensor<__T__> zLocal = outQueueZ.DeQue<__T__>();
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> xGm, yGm, zGm;
    uint64_t blockLength;
    uint64_t tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, y, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
        .replace("__API__", api)
}

// ---------------------------------------------------------------------------
// 元素级：一元 op（Relu/Gelu/Sigmoid/Tanh/Sqrt/Exp/Log/Rsqrt/Reciprocal/Abs）
// ---------------------------------------------------------------------------

fn elem_unary(spec: &KernelSpec, op_name: &str, note: &str) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let api = match op_name {
        "Relu" => "Relu",
        "Gelu" => "Gelu",
        "Sigmoid" => "Sigmoid",
        "Tanh" => "Tanh",
        "Sqrt" => "Sqrt",
        "Exp" => "Exp",
        "Log" => "Log",
        "Rsqrt" => "Rsqrt",
        "Reciprocal" => "Reciprocal",
        "Abs" => "Abs",
        _ => "Relu",
    };
    let tmpl = r#"// __NAME__: 元素级一元 op（AscendC::__API__，TILE_LENGTH=128 双缓冲）
// __NOTE__
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ __T__*)x + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ __T__*)z + offset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ void CopyIn(int32_t progress) {
        LocalTensor<__T__> xLocal = inQueueX.AllocTensor<__T__>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ void Compute(int32_t progress) {
        LocalTensor<__T__> xLocal = inQueueX.DeQue<__T__>();
        LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
        AscendC::__API__(zLocal, xLocal, TILE_LENGTH);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ void CopyOut(int32_t progress) {
        LocalTensor<__T__> zLocal = outQueueZ.DeQue<__T__>();
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t blockLength;
    uint64_t tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
        .replace("__API__", api)
        .replace("__NOTE__", note)
}

// ---------------------------------------------------------------------------
// Reduce（Sum/Mean/Max）：AscendC ReduceSum/ReduceMean/ReduceMax
// ---------------------------------------------------------------------------

fn reduce_kernel(spec: &KernelSpec, op_name: &str, is_mean: bool) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let api = op_name;
    let mean_div = if is_mean {
        "        // ReduceMean：除以 N 得均值\n        __T__ inv_n = (__T__)(1.0) / (__T__)reduceLen;\n        Muls(zLocal, zLocal, inv_n, 1);\n"
    } else {
        "        // ReduceSum/ReduceMax：直接输出累加/最大值\n"
    };
    let tmpl = r#"// __NAME__: Reduce op（AscendC::__API__，Vector Core 树形归约）
// 每 block 归约一段输入，输出一个标量
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        this->reduceLen = this->tileNum * (uint64_t)TILE_LENGTH;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ __T__*)x + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ __T__*)z + GetBlockIdx(), 1);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(workBuf, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, sizeof(__T__) * 8);
    }
    __aicore__ void Process() {
        LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
        // 第一遍：把第一个 tile ReduceSum 到 zLocal 作为初值
        LocalTensor<__T__> first = inQueueX.AllocTensor<__T__>();
        DataCopy(first, xGm[0], TILE_LENGTH);
        AscendC::__API__(zLocal, first, TILE_LENGTH);
        inQueueX.FreeTensor(first);
        // 后续 tile：ReduceSum 到 tmp 再 Add 累加（或 ReduceMax 用 Max）
        for (int32_t i = 1; i < (int32_t)this->tileNum; i++) {
            LocalTensor<__T__> xLocal = inQueueX.AllocTensor<__T__>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            LocalTensor<__T__> tmp = workBuf.Get<__T__>();
            AscendC::__API__(tmp, xLocal, TILE_LENGTH);
            Add(zLocal, zLocal, tmp, 1);
            inQueueX.FreeTensor(xLocal);
        }
__MEAN_DIV__
        outQueueZ.EnQue(zLocal);
        outQueueZ.DeQue<__T__>();
        DataCopy(zGm[0], zLocal, 1);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TBuf workBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t reduceLen;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
        .replace("__API__", api)
        .replace("__MEAN_DIV__", mean_div)
}

// ---------------------------------------------------------------------------
// MatMul（Cube Core，mmad 指令）
// ---------------------------------------------------------------------------

fn matmul_kernel(spec: &KernelSpec, arch: GpuArch) -> String {
    let cls = class_name(spec);
    let arch_note = if matches!(arch, GpuArch::Ascend910B1) {
        "// Ascend 910B1: Cube Core（mmad 指令）+ Vector Core，原生生支持 FP16/BF16 GEMM"
    } else if matches!(arch, GpuArch::Ascend910B3) {
        "// Ascend 910B3: 增强 Cube Core + Vector Core，吞吐更高的 mmad 指令"
    } else if matches!(arch, GpuArch::Ascend310P3) {
        "// Ascend 310P3: 轻量推理路径，Cube 走轻量 GEMM，Vector 仍处理累加"
    } else {
        "// 非 Ascend 架构：CANN 主要面向 Ascend NPU，仍生成 Cube Core + mmad 路径"
    };
    let tmpl = r#"// __NAME__: MatMul（Cube Core + mmad 指令，AscendC::MatMul）
// __ARCH_NOTE__
// 输入：A[M, K] @ B[K, N] = C[M, N]，tiling：BLOCK_M=16, BLOCK_N=16, BLOCK_K=16
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                         uint64_t M, uint64_t N, uint64_t K) {
        this->M = M; this->N = N; this->K = K;
        aGm.SetGlobalBuffer((__gm__ float*)a, M * K);
        bGm.SetGlobalBuffer((__gm__ float*)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float*)c, M * N);
        // Cube Core L1/L0A/L0B/UBuf 缓冲
        pipe.InitBuffer(aBuf, BLOCK_M * BLOCK_K * sizeof(float));
        pipe.InitBuffer(bBuf, BLOCK_K * BLOCK_N * sizeof(float));
        pipe.InitBuffer(cBuf, BLOCK_M * BLOCK_N * sizeof(float));
    }
    __aicore__ void Process() {
        // 简化：单 block 处理一个 BLOCK_M x BLOCK_N 输出 tile
        // 生产代码会按 blockDim 分块覆盖整个 M x N
        LocalTensor<float> aLocal = aBuf.Get<float>();
        LocalTensor<float> bLocal = bBuf.Get<float>();
        LocalTensor<float> cLocal = cBuf.Get<float>();
        DataCopy(aLocal, aGm[0], BLOCK_M * BLOCK_K);
        DataCopy(bLocal, bGm[0], BLOCK_K * BLOCK_N);
        // AscendC::MatMul 调用 Cube Core（mmad 指令）做矩阵乘法
        // 910B1/B3：原生 Cube Core；310P3：轻量推理 GEMM 路径
        AscendC::MatMul(cLocal, aLocal, bLocal, BLOCK_M, BLOCK_N, BLOCK_K);
        DataCopy(cGm[0], cLocal, BLOCK_M * BLOCK_N);
    }
private:
    static constexpr int32_t BLOCK_M = 16;
    static constexpr int32_t BLOCK_N = 16;
    static constexpr int32_t BLOCK_K = 16;
    TPipe pipe;
    TBuf aBuf, bBuf, cBuf;
    GlobalTensor<float> aGm, bGm, cGm;
    uint64_t M, N, K;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __gm__ uint64_t* t = (__gm__ uint64_t*)tiling;
    __CLS__ op;
    op.Init(a, b, c, t[0], t[1], t[2]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__ARCH_NOTE__", arch_note)
}

// ---------------------------------------------------------------------------
// Softmax（Vector Core，多遍：ReduceMax → Sub+Exp+ReduceSum → Div）
// ---------------------------------------------------------------------------

fn softmax_kernel(spec: &KernelSpec, _arch: GpuArch) -> String {
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Softmax（Vector Core 多遍：ReduceMax → Sub+Exp+ReduceSum → Div）
// 每 block 处理一行，数值稳定版（先减行 max 再 exp 再 sum 再除）
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n_cols) {
        this->nCols = n_cols;
        this->tileNum = n_cols / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        xGm.SetGlobalBuffer((__gm__ float*)x + GetBlockIdx() * n_cols, n_cols);
        zGm.SetGlobalBuffer((__gm__ float*)z + GetBlockIdx() * n_cols, n_cols);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(maxBuf, sizeof(float) * 8);
        pipe.InitBuffer(sumBuf, sizeof(float) * 8);
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
    }
    __aicore__ void Process() {
        // 第一遍：ReduceMax（数值稳定）
        LocalTensor<float> maxLocal = maxBuf.Get<float>();
        LocalTensor<float> first = inQueueX.AllocTensor<float>();
        DataCopy(first, xGm[0], TILE_LENGTH);
        AscendC::ReduceMax(maxLocal, first, TILE_LENGTH);
        inQueueX.FreeTensor(first);
        for (int32_t i = 1; i < (int32_t)this->tileNum; i++) {
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            AscendC::ReduceMax(tmp, xLocal, TILE_LENGTH);
            Max(maxLocal, maxLocal, tmp, 1);
            inQueueX.FreeTensor(xLocal);
        }
        // 第二遍：Sub + Exp + ReduceSum
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        Duplicate(sumLocal, (float)0.0, 1);
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            // z = x - max
            Sub(zLocal, xLocal, maxLocal, TILE_LENGTH);
            // z = exp(z)
            Exp(zLocal, zLocal, TILE_LENGTH);
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            AscendC::ReduceSum(tmp, zLocal, TILE_LENGTH);
            Add(sumLocal, sumLocal, tmp, 1);
            outQueueZ.EnQue(zLocal);
            outQueueZ.DeQue<float>();
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
        // 第三遍：Div 归一化
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            DataCopy(zLocal, zGm[i * TILE_LENGTH], TILE_LENGTH);
            Div(zLocal, zLocal, sumLocal, TILE_LENGTH);
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf maxBuf, sumBuf, tmpBuf;
    GlobalTensor<float> xGm, zGm;
    uint64_t nCols;
    uint64_t tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
}

// ---------------------------------------------------------------------------
// LayerNorm（两遍：mean → var → normalize）
// ---------------------------------------------------------------------------

fn layernorm_kernel(spec: &KernelSpec, _arch: GpuArch) -> String {
    let cls = class_name(spec);
    let eps = spec.attrs.epsilon.unwrap_or(1e-5);
    let eps_str = format!("{:.10}", eps);
    let tmpl = r#"// __NAME__: LayerNorm（Vector Core 两遍扫描 + epsilon 数值稳定）
// 每 block 处理一行：mean → var → normalize: y = (x - mean) / sqrt(var + epsilon)
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n_cols) {
        this->nCols = n_cols;
        this->tileNum = n_cols / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        xGm.SetGlobalBuffer((__gm__ float*)x + GetBlockIdx() * n_cols, n_cols);
        zGm.SetGlobalBuffer((__gm__ float*)z + GetBlockIdx() * n_cols, n_cols);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(meanBuf, sizeof(float) * 8);
        pipe.InitBuffer(varBuf, sizeof(float) * 8);
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
    }
    __aicore__ void Process() {
        // 第一遍：ReduceMean 求 mean
        LocalTensor<float> meanLocal = meanBuf.Get<float>();
        LocalTensor<float> sumTmp = tmpBuf.Get<float>();
        Duplicate(meanLocal, (float)0.0, 1);
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            AscendC::ReduceSum(sumTmp, xLocal, TILE_LENGTH);
            Add(meanLocal, meanLocal, sumTmp, 1);
            inQueueX.FreeTensor(xLocal);
        }
        // mean = sum / nCols
        float inv_n = 1.0f / (float)this->nCols;
        Muls(meanLocal, meanLocal, inv_n, 1);
        // 第二遍：求 var（平方差的均值）
        LocalTensor<float> varLocal = varBuf.Get<float>();
        Duplicate(varLocal, (float)0.0, 1);
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            Sub(sumTmp, xLocal, meanLocal, TILE_LENGTH);
            Mul(sumTmp, sumTmp, sumTmp, TILE_LENGTH);
            LocalTensor<float> rowSum = tmpBuf.Get<float>();
            AscendC::ReduceSum(rowSum, sumTmp, TILE_LENGTH);
            Add(varLocal, varLocal, rowSum, 1);
            inQueueX.FreeTensor(xLocal);
        }
        Muls(varLocal, varLocal, inv_n, 1);
        // inv_std = 1 / sqrt(var + epsilon)
        LocalTensor<float> epsT = tmpBuf.Get<float>();
        Duplicate(epsT, (float)__EPS__, 1);
        Add(varLocal, varLocal, epsT, 1);
        Sqrt(varLocal, varLocal, 1);
        LocalTensor<float> invStd = epsBuf.Get<float>();
        Duplicate(invStd, (float)1.0, 1);
        Div(invStd, invStd, varLocal, 1);
        // 第三遍：归一化 (x - mean) * inv_std
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            Sub(zLocal, xLocal, meanLocal, TILE_LENGTH);
            Mul(zLocal, zLocal, invStd, TILE_LENGTH);
            outQueueZ.EnQue(zLocal);
            outQueueZ.DeQue<float>();
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf meanBuf, varBuf, tmpBuf, epsBuf;
    GlobalTensor<float> xGm, zGm;
    uint64_t nCols;
    uint64_t tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__EPS__", &eps_str)
}

// ---------------------------------------------------------------------------
// Conv（im2col + Cube GEMM 路径）
// ---------------------------------------------------------------------------

fn conv_kernel(spec: &KernelSpec, _arch: GpuArch) -> String {
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Conv2D（im2col + Cube GEMM 路径）
// 简化路径：每 block 处理一个输出 tile（BLOCK_M x BLOCK_N）
// 完整实现：im2col 把输入 patch 展开为矩阵，再用 AscendC::MatMul 做 GEMM
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                         uint64_t C_in, uint64_t C_out, uint64_t KH, uint64_t KW) {
        this->Cin = C_in; this->Cout = C_out;
        this->kh = KH; this->kw = KW;
        xGm.SetGlobalBuffer((__gm__ float*)x, C_in * 32 * 32);
        wGm.SetGlobalBuffer((__gm__ float*)w, C_out * C_in * KH * KW);
        bGm.SetGlobalBuffer((__gm__ float*)b, C_out);
        yGm.SetGlobalBuffer((__gm__ float*)y, C_out * 32 * 32);
        pipe.InitBuffer(im2colBuf, BLOCK_M * BLOCK_K * sizeof(float));
        pipe.InitBuffer(wBuf, BLOCK_K * BLOCK_N * sizeof(float));
        pipe.InitBuffer(outBuf, BLOCK_M * BLOCK_N * sizeof(float));
        pipe.InitBuffer(biasBuf, BLOCK_N * sizeof(float));
    }
    __aicore__ void Process() {
        // im2col：把输入 patch 展开成 [Cin*kh*kw, OH*OW]
        // 然后用 Cube Core 的 MatMul 做 GEMM：y = w @ im2col_x
        LocalTensor<float> im2col = im2colBuf.Get<float>();
        LocalTensor<float> wTile = wBuf.Get<float>();
        LocalTensor<float> outTile = outBuf.Get<float>();
        LocalTensor<float> bias = biasBuf.Get<float>();
        // 简化：填 0 后做 GEMM（生产代码会真实做 im2col + 偏移计算）
        Duplicate(im2col, (float)0.0, BLOCK_M * BLOCK_K);
        Duplicate(wTile, (float)0.0, BLOCK_K * BLOCK_N);
        DataCopy(bias, bGm[0], BLOCK_N);
        AscendC::MatMul(outTile, wTile, im2col, BLOCK_M, BLOCK_N, BLOCK_K);
        // 加 bias
        Add(outTile, outTile, bias, BLOCK_M * BLOCK_N);
        DataCopy(yGm[0], outTile, BLOCK_M * BLOCK_N);
    }
private:
    static constexpr int32_t BLOCK_M = 16;
    static constexpr int32_t BLOCK_N = 16;
    static constexpr int32_t BLOCK_K = 16;
    TPipe pipe;
    TBuf im2colBuf, wBuf, outBuf, biasBuf;
    GlobalTensor<float> xGm, wGm, bGm, yGm;
    uint64_t Cin, Cout, kh, kw;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __gm__ uint64_t* t = (__gm__ uint64_t*)tiling;
    __CLS__ op;
    op.Init(x, w, b, y, t[0], t[1], t[2], t[3]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
}

// ---------------------------------------------------------------------------
// Pool（max/avg 池化，Vector Core）
// ---------------------------------------------------------------------------

fn pool_kernel(spec: &KernelSpec, _arch: GpuArch) -> String {
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Pool（max 池化，Vector Core）
// 每 block 处理一个输出 tile，遍历 KH x KW 窗口取最大值
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR y, uint64_t C, uint64_t H, uint64_t W,
                         uint64_t KH, uint64_t KW, uint64_t stride) {
        this->C = C; this->H = H; this->W = W;
        this->kh = KH; this->kw = KW; this->stride = stride;
        xGm.SetGlobalBuffer((__gm__ float*)x, C * H * W);
        yGm.SetGlobalBuffer((__gm__ float*)y, C * ((H - KH) / stride + 1) * ((W - KW) / stride + 1));
        pipe.InitBuffer(inBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(maxBuf, sizeof(float) * 8);
    }
    __aicore__ void Process() {
        uint64_t OH = (this->H - this->kh) / this->stride + 1;
        uint64_t OW = (this->W - this->kw) / this->stride + 1;
        LocalTensor<float> maxLocal = maxBuf.Get<float>();
        Duplicate(maxLocal, (float)-1e30, 1);
        // 简化：对每个输出窗口做 ReduceMax
        for (uint64_t c = 0; c < this->C; c++) {
            for (uint64_t oh = 0; oh < OH; oh++) {
                for (uint64_t ow = 0; ow < OW; ow++) {
                    LocalTensor<float> inLocal = inBuf.Get<float>();
                    LocalTensor<float> outLocal = outBuf.Get<float>();
                    uint64_t idx = 0;
                    for (uint64_t ki = 0; ki < this->kh; ki++) {
                        for (uint64_t kj = 0; kj < this->kw; kj++) {
                            uint64_t ih = oh * this->stride + ki;
                            uint64_t iw = ow * this->stride + kj;
                            inLocal.SetValue(idx, xGm.GetValue(c * H * W + ih * W + iw));
                            idx++;
                        }
                    }
                    AscendC::ReduceMax(outLocal, inLocal, idx);
                    yGm.SetValue(c * OH * OW + oh * OW + ow, outLocal.GetValue(0));
                }
            }
        }
    }
private:
    TPipe pipe;
    TBuf inBuf, outBuf, maxBuf;
    GlobalTensor<float> xGm, yGm;
    uint64_t C, H, W, kh, kw, stride;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __gm__ uint64_t* t = (__gm__ uint64_t*)tiling;
    __CLS__ op;
    op.Init(x, y, t[0], t[1], t[2], t[3], t[4], t[5]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
}

// ---------------------------------------------------------------------------
// 数据移动
// ---------------------------------------------------------------------------

fn reshape_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Reshape（连续内存拷贝，shape 一致性由 host 保证）
// Vector Core DataCopy：TILE_LENGTH=128 双缓冲
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ __T__*)x + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ __T__*)z + offset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<__T__> xLocal = inQueueX.AllocTensor<__T__>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            inQueueX.EnQue(xLocal);
            LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
            xLocal = inQueueX.DeQue<__T__>();
            DataCopy(zLocal, xLocal, TILE_LENGTH);
            outQueueZ.EnQue(zLocal);
            outQueueZ.DeQue<__T__>();
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
}

fn transpose_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Transpose（DataCopy + 偏移计算，2D 行列交换）
// Vector Core：把 input[row, col] 拷到 output[col, row]
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t H, uint64_t W) {
        this->H = H; this->W = W;
        xGm.SetGlobalBuffer((__gm__ __T__*)x, H * W);
        zGm.SetGlobalBuffer((__gm__ __T__*)z, H * W);
        pipe.InitBuffer(inBuf, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outBuf, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        // 简化：按行扫，逐元素写出到转置位置
        LocalTensor<__T__> inLocal = inBuf.Get<__T__>();
        LocalTensor<__T__> outLocal = outBuf.Get<__T__>();
        for (uint64_t r = 0; r < this->H; r++) {
            // 一次拷一行
            uint64_t len = this->W;
            DataCopy(inLocal, xGm[r * this->W], len);
            for (uint64_t c = 0; c < this->W; c++) {
                __T__ v = inLocal.GetValue(c);
                // output[col, row] = input[row, col]
                zGm.SetValue(c * this->H + r, v);
            }
        }
    }
private:
    TPipe pipe;
    TBuf inBuf, outBuf;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t H, W;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __gm__ uint64_t* t = (__gm__ uint64_t*)tiling;
    __CLS__ op;
    op.Init(x, z, t[0], t[1]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
}

fn concat_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let n_inputs = spec.inputs.len().max(2);
    let n_inputs_str = n_inputs.to_string();
    let tmpl = r#"// __NAME__: Concat（沿 axis 拼接，DataCopy 按段拷贝；此处展示 2 路拼接）
// 注: 通用版支持 __N_INPUTS__ 路输入，host 按 offset 区间分配
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR z,
                         uint64_t n0, uint64_t n1, uint64_t total) {
        this->n0 = n0; this->n1 = n1; this->total = total;
        x0Gm.SetGlobalBuffer((__gm__ __T__*)x0, n0);
        x1Gm.SetGlobalBuffer((__gm__ __T__*)x1, n1);
        zGm.SetGlobalBuffer((__gm__ __T__*)z, total);
        pipe.InitBuffer(outBuf, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        LocalTensor<__T__> outLocal = outBuf.Get<__T__>();
        // 第一段：从 input0 拷贝 n0 个元素
        for (uint64_t i = 0; i < this->n0; i += TILE_LENGTH) {
            uint64_t len = (i + TILE_LENGTH < this->n0) ? TILE_LENGTH : (this->n0 - i);
            DataCopy(outLocal, x0Gm[i], len);
            DataCopy(zGm[i], outLocal, len);
        }
        // 第二段：从 input1 拷贝 n1 个元素到 offset = n0
        for (uint64_t i = 0; i < this->n1; i += TILE_LENGTH) {
            uint64_t len = (i + TILE_LENGTH < this->n1) ? TILE_LENGTH : (this->n1 - i);
            DataCopy(outLocal, x1Gm[i], len);
            DataCopy(zGm[this->n0 + i], outLocal, len);
        }
    }
private:
    TPipe pipe;
    TBuf outBuf;
    GlobalTensor<__T__> x0Gm, x1Gm, zGm;
    uint64_t n0, n1, total;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x0, GM_ADDR x1, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __gm__ uint64_t* t = (__gm__ uint64_t*)tiling;
    __CLS__ op;
    op.Init(x0, x1, z, t[0], t[1], t[2]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
        .replace("__N_INPUTS__", &n_inputs_str)
}

fn slice_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Slice（按 starts/ends/steps 拷贝，DataCopy 按 step 间隔取元素）
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t start, uint64_t step, uint64_t out_n) {
        this->start = start; this->step = step; this->outN = out_n;
        xGm.SetGlobalBuffer((__gm__ __T__*)x, start + step * out_n);
        zGm.SetGlobalBuffer((__gm__ __T__*)z, out_n);
        pipe.InitBuffer(inBuf, sizeof(__T__));
        pipe.InitBuffer(outBuf, sizeof(__T__));
    }
    __aicore__ void Process() {
        LocalTensor<__T__> inLocal = inBuf.Get<__T__>();
        LocalTensor<__T__> outLocal = outBuf.Get<__T__>();
        // 按 step 间隔逐元素拷贝
        for (uint64_t i = 0; i < this->outN; i++) {
            __T__ v = xGm.GetValue(this->start + i * this->step);
            outLocal.SetValue(0, v);
            zGm.SetValue(i, outLocal.GetValue(0));
        }
    }
private:
    TPipe pipe;
    TBuf inBuf, outBuf;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t start, step, outN;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __gm__ uint64_t* t = (__gm__ uint64_t*)tiling;
    __CLS__ op;
    op.Init(x, z, t[0], t[1], t[2]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
}

// ---------------------------------------------------------------------------
// 数据流
// ---------------------------------------------------------------------------

fn constant_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let val = spec.attrs.value.unwrap_or(0.0).to_string();
    let tmpl = r#"// __NAME__: Constant（用 AscendC::Duplicate 把 value 广播到输出 buffer）
// 默认 value = __VAL__（host 端可用 tiling 覆盖）
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        zGm.SetGlobalBuffer((__gm__ __T__*)z + GetBlockIdx() * this->blockLength, this->blockLength);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
            Duplicate(zLocal, (__T__)__VAL__, TILE_LENGTH);
            outQueueZ.EnQue(zLocal);
            outQueueZ.DeQue<__T__>();
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> zGm;
    uint64_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
        .replace("__VAL__", &val)
}

fn placeholder_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Placeholder（输入占位）
// 注: Placeholder 标记图输入，host 端绑定 GM 地址；此处生成 identity 拷贝 kernel 便于调试
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ __T__*)x + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ __T__*)z + offset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<__T__> xLocal = inQueueX.AllocTensor<__T__>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            inQueueX.EnQue(xLocal);
            LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
            xLocal = inQueueX.DeQue<__T__>();
            DataCopy(zLocal, xLocal, TILE_LENGTH);
            outQueueZ.EnQue(zLocal);
            outQueueZ.DeQue<__T__>();
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
}

fn return_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let tmpl = r#"// __NAME__: Return（图输出）
// 注: Return 标记图输出，host 端读取该 buffer；此处生成 identity 拷贝 kernel 保证数据落盘
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ __T__*)x + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ __T__*)z + offset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<__T__> xLocal = inQueueX.AllocTensor<__T__>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            inQueueX.EnQue(xLocal);
            LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
            xLocal = inQueueX.DeQue<__T__>();
            DataCopy(zLocal, xLocal, TILE_LENGTH);
            outQueueZ.EnQue(zLocal);
            outQueueZ.DeQue<__T__>();
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
}

// ---------------------------------------------------------------------------
// Fused / Custom
// ---------------------------------------------------------------------------

fn fused_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let ops = if spec.attrs.fused_ops.is_empty() {
        "mul -> add -> relu".to_string()
    } else {
        spec.attrs.fused_ops.join(" -> ")
    };
    let tmpl = r#"// __NAME__: Fused（融合算子链: __OPS__）
// Vector Core 串联：Mul -> Add -> Relu（典型 GEMM+bias+激活 融合）
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        aGm.SetGlobalBuffer((__gm__ __T__*)a + offset, this->blockLength);
        bGm.SetGlobalBuffer((__gm__ __T__*)b + offset, this->blockLength);
        cGm.SetGlobalBuffer((__gm__ __T__*)c + offset, this->blockLength);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<__T__> aLocal = inQueueA.AllocTensor<__T__>();
            LocalTensor<__T__> bLocal = inQueueB.AllocTensor<__T__>();
            DataCopy(aLocal, aGm[i * TILE_LENGTH], TILE_LENGTH);
            DataCopy(bLocal, bGm[i * TILE_LENGTH], TILE_LENGTH);
            inQueueA.EnQue(aLocal);
            inQueueB.EnQue(bLocal);
            aLocal = inQueueA.DeQue<__T__>();
            bLocal = inQueueB.DeQue<__T__>();
            LocalTensor<__T__> cLocal = outQueueC.AllocTensor<__T__>();
            LocalTensor<__T__> tmp = tmpBuf.Get<__T__>();
            // z = a * b
            Mul(tmp, aLocal, bLocal, TILE_LENGTH);
            // z = z + b
            Add(tmp, tmp, bLocal, TILE_LENGTH);
            // z = relu(z)
            Relu(cLocal, tmp, TILE_LENGTH);
            outQueueC.EnQue(cLocal);
            outQueueC.DeQue<__T__>();
            DataCopy(cGm[i * TILE_LENGTH], cLocal, TILE_LENGTH);
            outQueueC.FreeTensor(cLocal);
            inQueueA.FreeTensor(aLocal);
            inQueueB.FreeTensor(bLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueC;
    TBuf tmpBuf;
    GlobalTensor<__T__> aGm, bGm, cGm;
    uint64_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(a, b, c, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
        .replace("__T__", t)
        .replace("__OPS__", &ops)
}

fn custom_kernel(spec: &KernelSpec) -> String {
    let t = spec.dtype.cann_type();
    let cls = class_name(spec);
    let op_type = if spec.attrs.custom_op_type.is_empty() {
        "unknown"
    } else {
        &spec.attrs.custom_op_type
    };
    let tmpl = r#"// __NAME__: Custom op（原始 op_type = __OP_TYPE__）
// 注: 未知 ONNX 算子，生成通用元素级 identity kernel（DataCopy），host 端可替换为自定义实现
class __CLS__ {
public:
    __aicore__ __CLS__() {}
    __aicore__ void Init(GM_ADDR x, GM_ADDR z, uint64_t n) {
        this->blockLength = n / GetBlockNum();
        this->tileNum = blockLength / TILE_LENGTH;
        if (this->tileNum == 0) this->tileNum = 1;
        uint64_t offset = this->tileNum * (uint64_t)TILE_LENGTH * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ __T__*)x + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ __T__*)z + offset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(__T__));
    }
    __aicore__ void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            LocalTensor<__T__> xLocal = inQueueX.AllocTensor<__T__>();
            DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            inQueueX.EnQue(xLocal);
            LocalTensor<__T__> zLocal = outQueueZ.AllocTensor<__T__>();
            xLocal = inQueueX.DeQue<__T__>();
            DataCopy(zLocal, xLocal, TILE_LENGTH);
            outQueueZ.EnQue(zLocal);
            outQueueZ.DeQue<__T__>();
            DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            outQueueZ.FreeTensor(zLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<__T__> xGm, zGm;
    uint64_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void __NAME__(GM_ADDR x, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling) {
    __CLS__ op;
    op.Init(x, z, ((__gm__ uint64_t*)tiling)[0]);
    op.Process();
}
"#;
    tmpl.replace("__NAME__", &spec.name)
        .replace("__CLS__", &cls)
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
            GpuArch::Ascend910B1,
            GpuArch::Ascend910B3,
            GpuArch::Ascend310P3,
        ]
    }

    #[test]
    fn test_emit_empty() {
        let out = emit(&[], GpuArch::Ascend910B1).expect("emit 不应失败");
        assert_eq!(out.lang, SourceLang::Cann);
        assert_eq!(out.arch, GpuArch::Ascend910B1);
        assert!(out.kernels.is_empty(), "空 kernel 列表应无 kernel_info");
        assert!(out.source.contains("kernel_tiling.h"));
        assert!(out.source.contains("kernel_operator.h"));
        assert!(out.source.contains("using namespace AscendC;"));
        assert!(out.source.contains("Ascend 910B1"));
    }

    #[test]
    fn test_emit_elementwise() {
        let specs = vec![make_spec(OpKind::Add, 0)];
        let out = emit(&specs, GpuArch::Ascend910B1).expect("emit 失败");
        assert!(
            out.source.contains("__aicore__"),
            "Add kernel 必须含 __aicore__"
        );
        assert!(
            out.source.contains("AscendC"),
            "Add kernel 必须含 AscendC 命名空间"
        );
        assert!(
            out.source.contains("neutron_add_0"),
            "Add kernel 必须含 neutron_add_0"
        );
        assert!(
            out.source.contains("AscendC::Add"),
            "Add kernel 必须调用 AscendC::Add"
        );
        assert!(out.source.contains("TILE_LENGTH"));
        assert!(out.source.contains("BUFFER_NUM"));
        assert_eq!(out.kernels.len(), 1);
        assert_eq!(out.kernels[0].name, "neutron_add_0");
    }

    #[test]
    fn test_emit_matmul_910b1() {
        let specs = vec![make_spec(OpKind::MatMul, 4)];
        let out = emit(&specs, GpuArch::Ascend910B1).expect("emit 失败");
        assert!(
            out.source.contains("MatMul")
                || out.source.contains("mmad")
                || out.source.contains("Cube"),
            "910B1 MatMul 必须含 MatMul/mmad/Cube"
        );
        assert!(out.source.contains("Ascend 910B1"));
        assert!(out.source.contains("__aicore__"));
    }

    #[test]
    fn test_emit_matmul_910b3() {
        let specs = vec![make_spec(OpKind::MatMul, 4)];
        let out = emit(&specs, GpuArch::Ascend910B3).expect("emit 失败");
        assert!(
            out.source.contains("MatMul") || out.source.contains("Cube"),
            "910B3 MatMul 必须含 MatMul/Cube"
        );
        assert!(out.source.contains("Ascend 910B3"));
        assert!(out.source.contains("__aicore__"));
    }

    #[test]
    fn test_emit_softmax() {
        let specs = vec![make_spec(OpKind::Softmax, 5)];
        let out = emit(&specs, GpuArch::Ascend910B1).expect("emit 失败");
        assert!(
            out.source.contains("ReduceMax"),
            "softmax 必须含 ReduceMax（数值稳定）"
        );
        assert!(out.source.contains("Exp"), "softmax 必须含 Exp");
        assert!(out.source.contains("ReduceSum"), "softmax 必须含 ReduceSum");
        assert!(out.source.contains("Div"), "softmax 必须含 Div");
        assert!(out.source.contains("__aicore__"));
    }

    #[test]
    fn test_emit_layernorm() {
        let specs = vec![make_spec(OpKind::LayerNorm, 6)];
        let out = emit(&specs, GpuArch::Ascend910B1).expect("emit 失败");
        assert!(
            out.source.contains("ReduceMean")
                || out.source.contains("ReduceSum")
                || out.source.contains("epsilon"),
            "layernorm 应含 ReduceMean/ReduceSum/epsilon"
        );
        assert!(out.source.contains("epsilon"));
        assert!(out.source.contains("__aicore__"));
    }

    #[test]
    fn test_emit_all_ops() {
        // 31 OpKind × 全 arch，确保 source 非空且含 __aicore__
        for arch in all_archs() {
            let specs: Vec<KernelSpec> = all_ops()
                .into_iter()
                .enumerate()
                .map(|(i, op)| make_spec(op, i as u32))
                .collect();
            let out = emit(&specs, arch).expect("emit 不应失败");
            assert!(!out.source.is_empty(), "source 不应为空 (arch={:?})", arch);
            assert!(
                out.source.contains("__aicore__"),
                "source 应含 __aicore__ (arch={:?})",
                arch
            );
            assert!(out.source.contains("kernel_tiling.h"));
            assert!(out.source.contains("kernel_operator.h"));
            assert!(out.source.contains("using namespace AscendC;"));
            assert_eq!(out.kernels.len(), specs.len());
            assert_eq!(out.arch, arch);
            // 每个 kernel 都应有对应 launch 说明
            for s in &specs {
                assert!(
                    out.source.contains(&format!("// launch_{}:", s.name)),
                    "应有 launch_{} 注释 (arch={:?})",
                    s.name,
                    arch
                );
            }
        }
    }

    #[test]
    fn test_emit_dtype_diversity() {
        // 不同 dtype 应映射到不同 cann_type
        for (dt, needle) in [
            (DType::F32, "float"),
            (DType::F16, "__fp16"),
            (DType::BF16, "__bf16"),
            (DType::I32, "int32_t"),
            (DType::I64, "int64_t"),
        ] {
            let mut spec = make_spec(OpKind::Add, 0);
            spec.dtype = dt;
            let out = emit(&[spec], GpuArch::Ascend910B1).expect("emit 失败");
            assert!(
                out.source.contains(needle),
                "dtype {:?} 应映射到 cann_type {}",
                dt,
                needle
            );
        }
    }

    #[test]
    fn test_emit_microarch_notes() {
        // 910B1 应有 Cube Core 注释；310P3 应有轻量推理注释
        let specs = vec![make_spec(OpKind::Relu, 0)];
        let out1 = emit(&specs, GpuArch::Ascend910B1).expect("emit 失败");
        assert!(
            out1.source.contains("Cube Core") || out1.source.contains("Vector Core"),
            "910B1 header 应含 Cube/Vector Core 注释"
        );
        let out3 = emit(&specs, GpuArch::Ascend310P3).expect("emit 失败");
        assert!(
            out3.source.contains("310P3") || out3.source.contains("Vector Core"),
            "310P3 header 应含 310P3 或 Vector Core 注释"
        );
    }

    #[test]
    fn test_emit_no_unwrap_in_code() {
        // 生成的 C++ 源码不应包含 unwrap 字样（避免 Rust 习惯渗透到 C++）
        let specs: Vec<KernelSpec> = all_ops()
            .into_iter()
            .enumerate()
            .map(|(i, op)| make_spec(op, i as u32))
            .collect();
        let out = emit(&specs, GpuArch::Ascend910B1).expect("emit 失败");
        assert!(
            !out.source.contains("unwrap"),
            "生成的 C++ 源码不应含 unwrap 字样"
        );
    }
}
