//! neutron IR 类型系统。

use std::fmt;

// ---- Dtype ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32, F16, F64,
    Int8, Int16, Int32, Int64,
    Bool,
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dtype::F32 => write!(f, "f32"), Dtype::F16 => write!(f, "f16"),
            Dtype::F64 => write!(f, "f64"), Dtype::Int8 => write!(f, "int8"),
            Dtype::Int16 => write!(f, "int16"), Dtype::Int32 => write!(f, "int32"),
            Dtype::Int64 => write!(f, "int64"), Dtype::Bool => write!(f, "bool"),
        }
    }
}

impl Dtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::Int32 => 4, Dtype::F16 | Dtype::Int16 => 2,
            Dtype::F64 | Dtype::Int64 => 8, Dtype::Int8 | Dtype::Bool => 1,
        }
    }
    pub fn is_floating(&self) -> bool { matches!(self, Dtype::F32 | Dtype::F16 | Dtype::F64) }
    pub fn is_integer(&self) -> bool { matches!(self, Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 | Dtype::Bool) }
}

// ---- Dim -----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    Static(usize), Dynamic(String), Unknown,
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Static(n) => write!(f, "{}", n),
            Dim::Dynamic(s) => write!(f, "{}", s),
            Dim::Unknown => write!(f, "?"),
        }
    }
}

impl Dim {
    pub fn named(name: impl Into<String>) -> Self { Dim::Dynamic(name.into()) }
    pub fn is_static(&self) -> bool { matches!(self, Dim::Static(_)) }
    pub fn static_value(&self) -> Option<usize> { match self { Dim::Static(n) => Some(*n), _ => None } }
}

// ---- Shape ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape { pub dims: Vec<Dim> }

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")
    }
}

impl Shape {
    pub fn new(dims: Vec<Dim>) -> Self { Shape { dims } }
    pub fn rank(&self) -> usize { self.dims.len() }
    pub fn is_static(&self) -> bool { self.dims.iter().all(|d| d.is_static()) }
    pub fn static_dims(&self) -> Option<Vec<usize>> { self.dims.iter().map(|d| d.static_value()).collect() }
    pub fn total_elements(&self) -> Option<usize> { self.static_dims().map(|d| d.iter().product()) }
}

// ---- TensorType ----------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType { pub dtype: Dtype, pub shape: Shape }

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}>{}", self.dtype, self.shape)
    }
}

impl TensorType {
    pub fn new(dtype: Dtype, dims: Vec<Dim>) -> Self { TensorType { dtype, shape: Shape::new(dims) } }
    pub fn rank(&self) -> usize { self.shape.rank() }
    pub fn size_bytes(&self) -> Option<usize> { self.shape.total_elements().map(|n| n * self.dtype.size_bytes()) }
}

// ---- ScalarType ----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarType { pub dtype: Dtype }

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.dtype) }
}

// ---- FunctionType --------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType { pub inputs: Vec<IrType>, pub outputs: Vec<IrType> }

impl fmt::Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, t) in self.inputs.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", t)?;
        }
        write!(f, ") -> (")?;
        for (i, t) in self.outputs.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", t)?;
        }
        write!(f, ")")
    }
}

// ---- IrType --------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    Tensor(TensorType), Scalar(ScalarType), Function(FunctionType), None,
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Tensor(t) => write!(f, "{}", t),
            IrType::Scalar(s) => write!(f, "{}", s),
            IrType::Function(ft) => write!(f, "fn{}", ft),
            IrType::None => write!(f, "none"),
        }
    }
}

impl IrType {
    pub fn tensor(dtype: Dtype, dims: Vec<Dim>) -> Self { IrType::Tensor(TensorType::new(dtype, dims)) }
    pub fn scalar(dtype: Dtype) -> Self { IrType::Scalar(ScalarType { dtype }) }
    pub fn func(inputs: Vec<IrType>, outputs: Vec<IrType>) -> Self { IrType::Function(FunctionType { inputs, outputs }) }
    pub fn is_tensor(&self) -> bool { matches!(self, IrType::Tensor(_)) }
    pub fn is_scalar(&self) -> bool { matches!(self, IrType::Scalar(_)) }
    pub fn is_none(&self) -> bool { matches!(self, IrType::None) }
    pub fn as_tensor(&self) -> Option<&TensorType> { match self { IrType::Tensor(t) => Some(t), _ => None } }
    pub fn as_scalar(&self) -> Option<&ScalarType> { match self { IrType::Scalar(s) => Some(s), _ => None } }
}

// ---- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_display() {
        assert_eq!(Dtype::F32.to_string(), "f32");
        assert_eq!(Dtype::Int8.to_string(), "int8");
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(Dtype::F32.size_bytes(), 4);
        assert_eq!(Dtype::F64.size_bytes(), 8);
        assert_eq!(Dtype::Int8.size_bytes(), 1);
    }

    #[test]
    fn test_shape_static() {
        let shape = Shape::new(vec![Dim::Static(2), Dim::Static(3)]);
        assert_eq!(shape.rank(), 2);
        assert!(shape.is_static());
        assert_eq!(shape.total_elements(), Some(6));
    }

    #[test]
    fn test_shape_dynamic() {
        let shape = Shape::new(vec![Dim::named("batch"), Dim::Static(256)]);
        assert_eq!(shape.rank(), 2);
        assert!(!shape.is_static());
        assert_eq!(shape.total_elements(), None);
    }

    #[test]
    fn test_tensor_type_display() {
        let t = TensorType::new(Dtype::F32, vec![Dim::named("batch"), Dim::Static(128)]);
        assert_eq!(t.to_string(), "Tensor<f32>(batch, 128)");
    }

    #[test]
    fn test_irtype_helpers() {
        let t = IrType::tensor(Dtype::F32, vec![Dim::Static(10)]);
        assert!(t.is_tensor());
        assert!(!t.is_scalar());
        let s = IrType::scalar(Dtype::Int32);
        assert!(s.is_scalar());
        assert_eq!(s.as_scalar().unwrap().dtype, Dtype::Int32);
    }
}