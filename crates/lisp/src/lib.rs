//! lisp — 独立 S-expr 解释器，无内部依赖。
//!
//! 用于 isel 的指令选择规则描述。

pub mod interp;
pub mod parser;

pub use interp::Interp;
pub use parser::parse;

/// S-expr 值
#[derive(Debug, Clone)]
pub enum Val {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Sym(String),
    Str(String),
    List(Vec<Val>),
    Lambda {
        params: Vec<String>,
        body: Vec<Val>,
    },
}

impl Val {
    pub fn as_list(&self) -> Option<&[Val]> {
        match self {
            Val::List(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_sym(&self) -> Option<&str> {
        match self {
            Val::Sym(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Val::Nil => write!(f, "nil"),
            Val::Bool(b) => write!(f, "{}", b),
            Val::Int(i) => write!(f, "{}", i),
            Val::Float(x) => write!(f, "{}", x),
            Val::Sym(s) => write!(f, "{}", s),
            Val::Str(s) => write!(f, "{}", s),
            Val::List(items) => {
                write!(f, "(")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            Val::Lambda { .. } => write!(f, "<lambda>"),
        }
    }
}
