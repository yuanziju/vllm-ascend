//! spec — 后端代码生成所需的核心规格类型
//!
//! 这些类型把 IR Graph 的节点信息提取成与后端无关的 KernelSpec，
//! 各后端（CUDA/Triton/Metal/CANN）从 KernelSpec 生成源码。

use base::DType;

// ---------------------------------------------------------------------------
// 微架构
// ---------------------------------------------------------------------------

/// 具体 GPU/NPU 微架构
///
/// 决定后端代码生成时的指令选择：
/// - CUDA：Hopper 用 wgmma + TMA，Blackwell 用 tensor memory，Ampere 用 mma.sync
/// - Metal：Apple6+ 用 simdgroup matrix
/// - CANN：Ascend910B 用 Vector + Cube Core
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArch {
    // --- NVIDIA ---
    /// Ampere SM80（A100）
    Ampere80,
    /// Hopper SM90（H100）— 支持 TMA + wgmma + async copy
    Hopper90,
    /// Blackwell SM100（B200）— 支持 FP4/FP6 + tensor memory
    Blackwell100,
    // --- Apple ---
    /// Apple M1 — simdgroup_matrix 8x8
    Apple6,
    /// Apple M2 — 增强 simdgroup
    Apple7,
    /// Apple M3 — 增强 GPU
    Apple8,
    // --- Ascend ---
    /// Ascend 910B1 — Vector Core + Cube Core
    Ascend910B1,
    /// Ascend 910B3 — 增强 Vector
    Ascend910B3,
    /// Ascend 310P3 — 轻量推理
    Ascend310P3,
}

impl GpuArch {
    /// CUDA compute capability (major, minor)
    pub fn sm_version(self) -> (u32, u32) {
        match self {
            GpuArch::Ampere80 => (8, 0),
            GpuArch::Hopper90 => (9, 0),
            GpuArch::Blackwell100 => (10, 0),
            _ => (8, 0), // 非 CUDA 架构默认返回 8.0
        }
    }

    /// 最大 shared memory（字节）
    pub fn max_shared_mem(self) -> usize {
        match self {
            GpuArch::Ampere80 => 164 * 1024, // 164KB
            GpuArch::Hopper90 => 228 * 1024, // 228KB
            GpuArch::Blackwell100 => 228 * 1024,
            GpuArch::Apple6 => 32 * 1024,
            GpuArch::Apple7 => 32 * 1024,
            GpuArch::Apple8 => 32 * 1024,
            GpuArch::Ascend910B1 => 192 * 1024, // Ascend Unified Buffer
            GpuArch::Ascend910B3 => 192 * 1024,
            GpuArch::Ascend310P3 => 64 * 1024,
        }
    }

    /// warp/wavefront/simd 大小
    pub fn warp_size(self) -> u32 {
        match self {
            GpuArch::Ampere80 | GpuArch::Hopper90 | GpuArch::Blackwell100 => 32,
            GpuArch::Apple6 | GpuArch::Apple7 | GpuArch::Apple8 => 32, // simdgroup = 32
            GpuArch::Ascend910B1 | GpuArch::Ascend910B3 | GpuArch::Ascend310P3 => 64, // Vector Core block
        }
    }

    /// 是否支持 TMA（Tensor Memory Accelerator）
    pub fn has_tma(self) -> bool {
        matches!(self, GpuArch::Hopper90 | GpuArch::Blackwell100)
    }

    /// 是否支持 wgmma（warp group MMA）
    pub fn has_wgmma(self) -> bool {
        matches!(self, GpuArch::Hopper90 | GpuArch::Blackwell100)
    }

    /// 是否支持 simdgroup_matrix
    pub fn has_simdgroup(self) -> bool {
        matches!(self, GpuArch::Apple6 | GpuArch::Apple7 | GpuArch::Apple8)
    }

    /// 是否支持 Cube Core（矩阵专用单元）
    pub fn has_cube_core(self) -> bool {
        matches!(self, GpuArch::Ascend910B1 | GpuArch::Ascend910B3)
    }

    pub fn name(self) -> &'static str {
        match self {
            GpuArch::Ampere80 => "Ampere SM80",
            GpuArch::Hopper90 => "Hopper SM90",
            GpuArch::Blackwell100 => "Blackwell SM100",
            GpuArch::Apple6 => "Apple M1",
            GpuArch::Apple7 => "Apple M2",
            GpuArch::Apple8 => "Apple M3",
            GpuArch::Ascend910B1 => "Ascend 910B1",
            GpuArch::Ascend910B3 => "Ascend 910B3",
            GpuArch::Ascend310P3 => "Ascend 310P3",
        }
    }
}

// ---------------------------------------------------------------------------
// 张量规格
// ---------------------------------------------------------------------------

/// 张量元信息：名字 + shape + dtype + 是否输入
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub dims: Vec<i64>,
    pub dtype: DType,
    pub is_input: bool,
}

impl TensorSpec {
    pub fn element_count(&self) -> usize {
        self.dims
            .iter()
            .filter(|d| **d > 0)
            .map(|d| *d as usize)
            .product()
    }

    pub fn bytes(&self) -> usize {
        self.element_count() * self.dtype.size_bytes()
    }

    /// 形状字符串，如 "1024, 1024"
    pub fn dims_str(&self) -> String {
        self.dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// C 风格类型名
    pub fn c_type(&self) -> &'static str {
        self.dtype.c_type()
    }
}

// ---------------------------------------------------------------------------
// 核属性
// ---------------------------------------------------------------------------

/// 算子属性（axis / perm / shape / strides / epsilon / constant value 等）
///
/// 各后端按需读取需要的字段。
#[derive(Debug, Clone, Default)]
pub struct KernelAttrs {
    /// Reduce / Concat / Softmax 的轴
    pub axis: Option<i64>,
    /// Transpose 的置换
    pub perm: Vec<i64>,
    /// Reshape / Fused 的目标 shape
    pub shape: Vec<i64>,
    /// Fused 的 side input strides
    pub strides: Vec<i64>,
    /// LayerNorm epsilon
    pub epsilon: Option<f64>,
    /// Constant 标量值
    pub value: Option<f64>,
    /// Constant 张量数据（多元素）
    pub tensor_data: Vec<f64>,
    /// Conv: stride/padding/dilation/groups
    pub conv_stride: Vec<i64>,
    pub conv_padding: Vec<i64>,
    pub conv_dilation: Vec<i64>,
    pub conv_groups: Option<i64>,
    /// Fused op 序列（op kind names）
    pub fused_ops: Vec<String>,
    /// Custom op 原始 op_type
    pub custom_op_type: String,
    /// Slice: starts/ends/axes/steps
    pub slice_starts: Vec<i64>,
    pub slice_ends: Vec<i64>,
    pub slice_axes: Vec<i64>,
    pub slice_steps: Vec<i64>,
    /// Pool: kernel_size/strides/padding
    pub pool_kernel: Vec<i64>,
    pub pool_stride: Vec<i64>,
    pub pool_padding: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Kernel 规格
// ---------------------------------------------------------------------------

/// 单个 kernel 的完整规格
///
/// 一个 KernelSpec 对应 IR Graph 中的一个计算节点（OpKind）。
/// 后端代码生成器遍历 KernelSpec 列表，为每个生成对应的 kernel 函数。
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// kernel 名字（唯一，如 "neutron_add_0"）
    pub name: String,
    /// 算子类型
    pub op: base::OpKind,
    /// 输入张量
    pub inputs: Vec<TensorSpec>,
    /// 输出张量
    pub outputs: Vec<TensorSpec>,
    /// 算子属性
    pub attrs: KernelAttrs,
    /// 默认 dtype（取首个输出的 dtype）
    pub dtype: DType,
    /// IR 图节点序号（调试用）
    pub node_idx: u32,
}

impl KernelSpec {
    /// 推断 launch 配置：grid + block + shared_mem
    ///
    /// elementwise 算子：1D grid，每 block 256 线程
    /// reduce：2D grid（每 reduce 轴一组）
    /// MatMul/Conv：2D/3D tiled grid
    pub fn launch(&self, arch: GpuArch) -> LaunchSpec {
        let _warp = arch.warp_size();
        match self.op {
            // elementwise：每线程处理一个元素
            base::OpKind::Add
            | base::OpKind::Sub
            | base::OpKind::Mul
            | base::OpKind::Div
            | base::OpKind::Relu
            | base::OpKind::Gelu
            | base::OpKind::Sigmoid
            | base::OpKind::Tanh
            | base::OpKind::Sqrt
            | base::OpKind::Exp
            | base::OpKind::Pow
            | base::OpKind::Rsqrt
            | base::OpKind::Reciprocal
            | base::OpKind::Abs
            | base::OpKind::Log => {
                let n = self.first_output_len();
                let block = 256u32;
                let grid = (n as u32).div_ceil(block).max(1);
                LaunchSpec {
                    grid: (grid, 1, 1),
                    block: (block, 1, 1),
                    shared_mem: 0,
                }
            }
            // reduce：每 block 算一个 reduce 输出
            base::OpKind::ReduceSum | base::OpKind::ReduceMean | base::OpKind::ReduceMax => {
                let block = 256u32;
                let n = self.first_output_len().max(1) as u32;
                LaunchSpec {
                    grid: (n, 1, 1),
                    block: (block, 1, 1),
                    shared_mem: block * 4, // 256 * sizeof(float)
                }
            }
            // MatMul：tiled，block = (32, 32, 1) 或 (16, 16, 1) 取决于 arch
            base::OpKind::MatMul => {
                if arch.has_wgmma() {
                    // Hopper: wgmma 64x256
                    LaunchSpec {
                        grid: (8, 8, 1),
                        block: (128, 1, 1), // warpgroup
                        shared_mem: arch.max_shared_mem() as u32,
                    }
                } else {
                    // Ampere/Metal: 16x16 tile
                    LaunchSpec {
                        grid: (8, 8, 1),
                        block: (16, 16, 1),
                        shared_mem: 16 * 16 * 4 * 2, // 2 tiles
                    }
                }
            }
            // Softmax/LayerNorm：每行一个 block
            base::OpKind::Softmax | base::OpKind::LayerNorm => {
                let rows = self
                    .outputs
                    .first()
                    .and_then(|t| t.dims.first().copied())
                    .unwrap_or(1) as u32;
                let block = 256u32;
                LaunchSpec {
                    grid: (rows.max(1), 1, 1),
                    block: (block, 1, 1),
                    shared_mem: block * 4,
                }
            }
            // Conv：im2col + GEMM
            base::OpKind::Conv => {
                let block = 16u32;
                LaunchSpec {
                    grid: (16, 16, 1),
                    block: (block, block, 1),
                    shared_mem: arch.max_shared_mem() as u32 / 2,
                }
            }
            // Pool
            base::OpKind::Pool => {
                let block = 256u32;
                let n = self.first_output_len();
                let grid = (n as u32).div_ceil(block).max(1);
                LaunchSpec {
                    grid: (grid, 1, 1),
                    block: (block, 1, 1),
                    shared_mem: 0,
                }
            }
            // 数据移动：elementwise
            base::OpKind::Reshape
            | base::OpKind::Transpose
            | base::OpKind::Concat
            | base::OpKind::Slice => {
                let n = self.first_output_len();
                let block = 256u32;
                let grid = (n as u32).div_ceil(block).max(1);
                LaunchSpec {
                    grid: (grid, 1, 1),
                    block: (block, 1, 1),
                    shared_mem: 0,
                }
            }
            // Fused：elementwise 链
            base::OpKind::Fused | base::OpKind::Custom => {
                let n = self.first_output_len();
                let block = 256u32;
                let grid = (n as u32).div_ceil(block).max(1);
                LaunchSpec {
                    grid: (grid, 1, 1),
                    block: (block, 1, 1),
                    shared_mem: 0,
                }
            }
            // Constant / Placeholder / Return：不需要 launch
            base::OpKind::Constant | base::OpKind::Placeholder | base::OpKind::Return => {
                LaunchSpec {
                    grid: (1, 1, 1),
                    block: (1, 1, 1),
                    shared_mem: 0,
                }
            }
        }
    }

    fn first_output_len(&self) -> usize {
        self.outputs.first().map(|t| t.element_count()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Launch 配置
// ---------------------------------------------------------------------------

/// kernel launch 配置
#[derive(Debug, Clone, Copy)]
pub struct LaunchSpec {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

impl LaunchSpec {
    pub fn grid_str(&self) -> String {
        format!("dim3({}, {}, {})", self.grid.0, self.grid.1, self.grid.2)
    }
    pub fn block_str(&self) -> String {
        format!("dim3({}, {}, {})", self.block.0, self.block.1, self.block.2)
    }
}

// ---------------------------------------------------------------------------
// 后端输出
// ---------------------------------------------------------------------------

/// 源码语言
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLang {
    Cuda,   // .cu
    Triton, // .py
    Metal,  // .metal (MSL)
    Cann,   // .cpp
}

impl SourceLang {
    pub fn extension(self) -> &'static str {
        match self {
            SourceLang::Cuda => "cu",
            SourceLang::Triton => "py",
            SourceLang::Metal => "metal",
            SourceLang::Cann => "cpp",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            SourceLang::Cuda => "CUDA C++",
            SourceLang::Triton => "Triton (Python)",
            SourceLang::Metal => "Metal Shading Language",
            SourceLang::Cann => "Ascend CANN C++",
        }
    }
}

/// 单个 kernel 的生成信息
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub name: String,
    pub launch: LaunchSpec,
    pub shared_mem: u32,
}

/// 后端代码生成输出
#[derive(Debug, Clone)]
pub struct BackendOutput {
    /// 完整源码字符串
    pub source: String,
    /// 源码语言
    pub lang: SourceLang,
    /// 每个kernel的信息（用于 launch 调用代码）
    pub kernels: Vec<KernelInfo>,
    /// 微架构
    pub arch: GpuArch,
}

// ---------------------------------------------------------------------------
// DType 辅助（trait 扩展，不能给外部 crate 的类型加 inherent impl）
// ---------------------------------------------------------------------------

/// DType 后端代码生成辅助 trait
pub trait DTypeExt {
    fn c_type(self) -> &'static str;
    fn msl_type(self) -> &'static str;
    fn triton_type(self) -> &'static str;
    fn cann_type(self) -> &'static str;
    fn size_bytes(self) -> usize;
}

impl DTypeExt for DType {
    fn c_type(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F16 => "__half",
            DType::BF16 => "__nv_bfloat16",
            DType::I64 => "long long",
            DType::I32 => "int",
            DType::Bool => "bool",
        }
    }

    fn msl_type(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F16 => "half",
            DType::BF16 => "bfloat",
            DType::I64 => "long",
            DType::I32 => "int",
            DType::Bool => "bool",
        }
    }

    fn triton_type(self) -> &'static str {
        match self {
            DType::F32 => "tl.float32",
            DType::F16 => "tl.float16",
            DType::BF16 => "tl.bfloat16",
            DType::I64 => "tl.int64",
            DType::I32 => "tl.int32",
            DType::Bool => "tl.int1",
        }
    }

    fn cann_type(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F16 => "__fp16",
            DType::BF16 => "__bf16",
            DType::I64 => "int64_t",
            DType::I32 => "int32_t",
            DType::Bool => "bool",
        }
    }

    fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::Bool => 2,
            DType::I64 => 8,
        }
    }
}
