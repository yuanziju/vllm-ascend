//! Op trait + SSA 值类型。

use super::ty::IrType;
use std::fmt;

/// Op 的唯一标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub u32);

/// SSA 值 — Op 的产出
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// Op 操作数 — 引用一个 SSA 值
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpOperand {
    pub value: ValueId,
    pub owner: OpId,
}

/// 所有算子的基础 trait
pub trait Op: fmt::Debug {
    fn name(&self) -> &str;
    fn dialect(&self) -> &str;
    fn inputs(&self) -> &[OpOperand];
    fn inputs_mut(&mut self) -> &mut [OpOperand];
    fn outputs(&self) -> &[ValueId];
    fn result_types(&self) -> Vec<IrType>;
    fn verify(&self) -> Result<(), String> { Ok(()) }
    fn id(&self) -> OpId;
    fn set_id(&mut self, id: OpId);
    fn input_count(&self) -> usize { self.inputs().len() }
    fn output_count(&self) -> usize { self.outputs().len() }

    /// CSE 去重 key: (dialect.name, Vec<输入 ValueId>)
    fn op_key(&self) -> (String, Vec<ValueId>) {
        (
            format!("{}.{}", self.dialect(), self.name()),
            self.inputs().iter().map(|o| o.value).collect(),
        )
    }

    /// 深拷贝（用于 Box<dyn Op> 的 Clone）
    fn clone_box(&self) -> Box<dyn Op>;

    /// Downcast 支持（用于 pass 中访问具体 op 的字段）
    fn as_any(&self) -> &dyn std::any::Any;
}