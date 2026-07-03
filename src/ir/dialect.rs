//! Dialect trait + 所有算子 struct 声明 + DialectRegistry。

use std::fmt;
use super::op::Op;
use crate::graph::ir::RegionId;

// ============================================================================
// Dialect trait
// ============================================================================

pub trait Dialect: fmt::Debug {
    fn name(&self) -> &str;
    fn verify_op(&self, _op: &dyn Op) -> Result<(), String> { Ok(()) }
}

// ============================================================================
// DialectRegistry
// ============================================================================

#[derive(Debug, Default)]
pub struct DialectRegistry {
    dialects: Vec<Box<dyn Dialect>>,
}

impl DialectRegistry {
    pub fn new() -> Self { DialectRegistry { dialects: Vec::new() } }
    pub fn register(&mut self, dialect: impl Dialect + 'static) {
        self.dialects.push(Box::new(dialect));
    }
}

// ============================================================================
// ArithDialect
// ============================================================================

#[derive(Debug)]
pub struct ArithDialect;
impl Dialect for ArithDialect { fn name(&self) -> &str { "arith" } }

macro_rules! arith_binary_op {
    ($name:ident, $op_name:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            pub id: super::op::OpId,
            pub operands: Vec<super::op::OpOperand>,
            pub result: super::op::ValueId,
            pub result_type: super::ty::IrType,
        }
        impl Op for $name {
            fn name(&self) -> &str { $op_name }
            fn dialect(&self) -> &str { "arith" }
            fn inputs(&self) -> &[super::op::OpOperand] { &self.operands }
            fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut self.operands }
            fn outputs(&self) -> &[super::op::ValueId] { std::slice::from_ref(&self.result) }
            fn result_types(&self) -> Vec<super::ty::IrType> { vec![self.result_type.clone()] }
            fn id(&self) -> super::op::OpId { self.id }
            fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
            fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }
    };
}

arith_binary_op!(AddOp, "add");
arith_binary_op!(SubOp, "sub");
arith_binary_op!(MulOp, "mul");
arith_binary_op!(DivOp, "div");
arith_binary_op!(PowOp, "pow");

macro_rules! arith_unary_op {
    ($name:ident, $op_name:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            pub id: super::op::OpId,
            pub operands: Vec<super::op::OpOperand>,
            pub result: super::op::ValueId,
            pub result_type: super::ty::IrType,
        }
        impl Op for $name {
            fn name(&self) -> &str { $op_name }
            fn dialect(&self) -> &str { "arith" }
            fn inputs(&self) -> &[super::op::OpOperand] { &self.operands }
            fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut self.operands }
            fn outputs(&self) -> &[super::op::ValueId] { std::slice::from_ref(&self.result) }
            fn result_types(&self) -> Vec<super::ty::IrType> { vec![self.result_type.clone()] }
            fn id(&self) -> super::op::OpId { self.id }
            fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
            fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }
    };
}

arith_unary_op!(SqrtOp, "sqrt");
arith_unary_op!(ExpOp, "exp");
arith_unary_op!(LogOp, "log");
arith_unary_op!(NegOp, "neg");

/// 常量值 — 用于常数折叠
#[derive(Debug, Clone)]
pub enum ConstantValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl ConstantValue {
    pub fn is_zero(&self) -> bool {
        match self {
            ConstantValue::F32(v) => *v == 0.0,
            ConstantValue::F64(v) => *v == 0.0,
            ConstantValue::I32(v) => *v == 0,
            ConstantValue::I64(v) => *v == 0,
            ConstantValue::Bool(v) => !*v,
        }
    }
    pub fn is_one(&self) -> bool {
        match self {
            ConstantValue::F32(v) => (*v - 1.0).abs() < f32::EPSILON,
            ConstantValue::F64(v) => (*v - 1.0).abs() < f64::EPSILON,
            ConstantValue::I32(v) => *v == 1,
            ConstantValue::I64(v) => *v == 1,
            ConstantValue::Bool(v) => *v,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstantOp {
    pub id: super::op::OpId,
    pub value: ConstantValue,
    pub result: super::op::ValueId,
    pub result_type: super::ty::IrType,
}
impl Op for ConstantOp {
    fn name(&self) -> &str { "constant" }
    fn dialect(&self) -> &str { "arith" }
    fn inputs(&self) -> &[super::op::OpOperand] { &[] }
    fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut [] }
    fn outputs(&self) -> &[super::op::ValueId] { std::slice::from_ref(&self.result) }
    fn result_types(&self) -> Vec<super::ty::IrType> { vec![self.result_type.clone()] }
    fn id(&self) -> super::op::OpId { self.id }
    fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
    fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

// ============================================================================
// TensorDialect
// ============================================================================

#[derive(Debug)]
pub struct TensorDialect;
impl Dialect for TensorDialect { fn name(&self) -> &str { "tensor" } }

#[derive(Debug, Clone)]
pub struct MatMulOp {
    pub id: super::op::OpId,
    pub operands: Vec<super::op::OpOperand>,
    pub result: super::op::ValueId,
    pub result_type: super::ty::IrType,
}
impl Op for MatMulOp {
    fn name(&self) -> &str { "matmul" }
    fn dialect(&self) -> &str { "tensor" }
    fn inputs(&self) -> &[super::op::OpOperand] { &self.operands }
    fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut self.operands }
    fn outputs(&self) -> &[super::op::ValueId] { std::slice::from_ref(&self.result) }
    fn result_types(&self) -> Vec<super::ty::IrType> { vec![self.result_type.clone()] }
    fn id(&self) -> super::op::OpId { self.id }
    fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
    fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

// ============================================================================
// ReduceDialect
// ============================================================================

#[derive(Debug)]
pub struct ReduceDialect;
impl Dialect for ReduceDialect { fn name(&self) -> &str { "reduce" } }

macro_rules! reduce_op {
    ($name:ident, $op_name:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            pub id: super::op::OpId,
            pub operands: Vec<super::op::OpOperand>,
            pub result: super::op::ValueId,
            pub result_type: super::ty::IrType,
            pub dim: i32,
            pub keepdim: bool,
        }
        impl Op for $name {
            fn name(&self) -> &str { $op_name }
            fn dialect(&self) -> &str { "reduce" }
            fn inputs(&self) -> &[super::op::OpOperand] { &self.operands }
            fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut self.operands }
            fn outputs(&self) -> &[super::op::ValueId] { std::slice::from_ref(&self.result) }
            fn result_types(&self) -> Vec<super::ty::IrType> { vec![self.result_type.clone()] }
            fn id(&self) -> super::op::OpId { self.id }
            fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
            fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }
    };
}

reduce_op!(ReduceSumOp, "reduce_sum");
reduce_op!(ReduceMeanOp, "reduce_mean");
reduce_op!(ReduceMaxOp, "reduce_max");
reduce_op!(ReduceMinOp, "reduce_min");

// ============================================================================
// ActivateDialect
// ============================================================================

#[derive(Debug)]
pub struct ActivateDialect;
impl Dialect for ActivateDialect { fn name(&self) -> &str { "activate" } }

macro_rules! activate_op {
    ($name:ident, $op_name:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            pub id: super::op::OpId,
            pub operands: Vec<super::op::OpOperand>,
            pub result: super::op::ValueId,
            pub result_type: super::ty::IrType,
        }
        impl Op for $name {
            fn name(&self) -> &str { $op_name }
            fn dialect(&self) -> &str { "activate" }
            fn inputs(&self) -> &[super::op::OpOperand] { &self.operands }
            fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut self.operands }
            fn outputs(&self) -> &[super::op::ValueId] { std::slice::from_ref(&self.result) }
            fn result_types(&self) -> Vec<super::ty::IrType> { vec![self.result_type.clone()] }
            fn id(&self) -> super::op::OpId { self.id }
            fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
            fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }
    };
}

activate_op!(ReluOp, "relu");
activate_op!(GeluOp, "gelu");
activate_op!(SigmoidOp, "sigmoid");
activate_op!(SwishOp, "swish");
activate_op!(TanhOp, "tanh");

// ============================================================================
// NormDialect
// ============================================================================

#[derive(Debug)]
pub struct NormDialect;
impl Dialect for NormDialect { fn name(&self) -> &str { "norm" } }

#[derive(Debug, Clone)]
pub struct LayerNormOp {
    pub id: super::op::OpId,
    pub operands: Vec<super::op::OpOperand>,
    pub result: super::op::ValueId,
    pub result_type: super::ty::IrType,
    pub eps: f32,
}
impl Op for LayerNormOp {
    fn name(&self) -> &str { "layer_norm" }
    fn dialect(&self) -> &str { "norm" }
    fn inputs(&self) -> &[super::op::OpOperand] { &self.operands }
    fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut self.operands }
    fn outputs(&self) -> &[super::op::ValueId] { std::slice::from_ref(&self.result) }
    fn result_types(&self) -> Vec<super::ty::IrType> { vec![self.result_type.clone()] }
    fn id(&self) -> super::op::OpId { self.id }
    fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
    fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

// ============================================================================
// ControlDialect
// ============================================================================

#[derive(Debug)]
pub struct ControlDialect;
impl Dialect for ControlDialect { fn name(&self) -> &str { "control" } }

#[derive(Debug, Clone)]
pub struct BranchOp {
    pub id: super::op::OpId,
    pub target: RegionId,
    pub args: Vec<super::op::ValueId>,
}
impl Op for BranchOp {
    fn name(&self) -> &str { "br" }
    fn dialect(&self) -> &str { "control" }
    fn inputs(&self) -> &[super::op::OpOperand] { &[] }
    fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut [] }
    fn outputs(&self) -> &[super::op::ValueId] { &[] }
    fn result_types(&self) -> Vec<super::ty::IrType> { vec![] }
    fn id(&self) -> super::op::OpId { self.id }
    fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
    fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[derive(Debug, Clone)]
pub struct ReturnOp {
    pub id: super::op::OpId,
    pub value: Option<super::op::ValueId>,
}
impl Op for ReturnOp {
    fn name(&self) -> &str { "return" }
    fn dialect(&self) -> &str { "control" }
    fn inputs(&self) -> &[super::op::OpOperand] { &[] }
    fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut [] }
    fn outputs(&self) -> &[super::op::ValueId] { &[] }
    fn result_types(&self) -> Vec<super::ty::IrType> { vec![] }
    fn id(&self) -> super::op::OpId { self.id }
    fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
    fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[derive(Debug, Clone)]
pub struct CallOp {
    pub id: super::op::OpId,
    pub callee: String,
    pub args: Vec<super::op::ValueId>,
    pub results: Vec<super::op::ValueId>,
    pub result_types: Vec<super::ty::IrType>,
}
impl Op for CallOp {
    fn name(&self) -> &str { "call" }
    fn dialect(&self) -> &str { "control" }
    fn inputs(&self) -> &[super::op::OpOperand] { &[] }
    fn inputs_mut(&mut self) -> &mut [super::op::OpOperand] { &mut [] }
    fn outputs(&self) -> &[super::op::ValueId] { &self.results }
    fn result_types(&self) -> Vec<super::ty::IrType> { self.result_types.clone() }
    fn id(&self) -> super::op::OpId { self.id }
    fn set_id(&mut self, id: super::op::OpId) { self.id = id; }
    fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}