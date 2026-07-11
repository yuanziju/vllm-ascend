//! arch — 架构描述 + lowering + 目标特定优化

pub mod cuda;
pub mod lowering;
pub mod npu;

use base::{Graph, Result, ValueId};

/// 设备描述（按 target 分派，存寄存器/存储参数供寄存器分配读）
#[derive(Debug, Clone, Copy)]
pub enum DeviceDesc {
    Cuda(cuda::CudaDesc),
    Npu(npu::NpuDesc),
    /// CPU 暂无特化描述
    Cpu,
}

impl Default for DeviceDesc {
    fn default() -> Self {
        Self::Cuda(cuda::CudaDesc::default())
    }
}

/// 架构图（lowering 后）
#[derive(Debug, Default)]
pub struct ArchGraph {
    pub ops: Vec<ArchOp>,
    pub target: common::Target,
    /// 设备描述（按 target 在 new 时初始化）
    pub desc: DeviceDesc,
}

/// 架构操作（lowering 后的算子）。携带 operand 让寄存器分配能读 def-use。
#[derive(Debug, Clone)]
pub enum ArchOp {
    /// KernelCall 携带 kernel 名 + 输入 ValueId 列表 + 输出 ValueId 列表
    KernelCall {
        name: String,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    },
    /// Load 携带目标地址 value 和输出 value
    Load { addr: ValueId, dst: ValueId },
    /// Store 携带地址 value 和源 value
    Store { addr: ValueId, src: ValueId },
}

impl ArchGraph {
    pub fn new(target: common::Target) -> Self {
        let desc = match target {
            common::Target::Cuda => DeviceDesc::Cuda(cuda::CudaDesc::default()),
            common::Target::Npu => DeviceDesc::Npu(npu::NpuDesc::default()),
            common::Target::Cpu => DeviceDesc::Cpu,
        };
        Self {
            ops: Vec::new(),
            target,
            desc,
        }
    }
    pub fn add(&mut self, op: ArchOp) {
        self.ops.push(op);
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// lowering 入口
pub fn lower(graph: &Graph, target: common::Target) -> Result<ArchGraph> {
    lowering::lower(graph, target)
}
