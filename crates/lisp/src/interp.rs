//! interp — 求值器

use crate::Val;
use std::collections::HashMap;

pub struct Interp {
    pub vars: HashMap<String, Val>,
}

impl Interp {
    pub fn new() -> Self {
        let mut vars = HashMap::new();
        vars.insert("nil".into(), Val::Nil);
        Self { vars }
    }

    pub fn eval(&mut self, v: &Val) -> Result<Val, String> {
        match v {
            Val::Nil | Val::Bool(_) | Val::Int(_) | Val::Float(_) | Val::Str(_) | Val::Lambda { .. } => {
                Ok(v.clone())
            }
            Val::Sym(s) => self
                .vars
                .get(s)
                .cloned()
                .ok_or_else(|| format!("未绑定符号: {}", s)),
            Val::List(items) => {
                if items.is_empty() {
                    return Ok(Val::List(vec![]));
                }
                let head = items[0].as_sym().ok_or("调用头部必须是符号")?;
                // 特殊形式
                match head {
                    "quote" => Ok(items.get(1).cloned().unwrap_or(Val::Nil)),
                    "if" => {
                        let cond = self.eval(items.get(1).ok_or("if 缺少条件")?)?;
                        let is_true = !matches!(cond, Val::Bool(false) | Val::Nil);
                        if is_true {
                            self.eval(items.get(2).ok_or("if 缺少真分支")?)
                        } else {
                            items
                                .get(3)
                                .map(|v| self.eval(v))
                                .unwrap_or(Ok(Val::Nil))
                        }
                    }
                    "do" => {
                        let mut result = Val::Nil;
                        for item in &items[1..] {
                            result = self.eval(item)?;
                        }
                        Ok(result)
                    }
                    _ => {
                        // 函数应用
                        let args: Vec<Val> = items[1..]
                            .iter()
                            .map(|v| self.eval(v))
                            .collect::<Result<_, _>>()?;
                        call_builtin(head, &args)
                    }
                }
            }
        }
    }
}

impl Default for Interp {
    fn default() -> Self {
        Self::new()
    }
}

fn call_builtin(name: &str, args: &[Val]) -> Result<Val, String> {
    match name {
        "+" => {
            let mut acc_int: i64 = 0;
            let mut acc_float: f64 = 0.0;
            let mut is_float = false;
            for a in args {
                match a {
                    Val::Int(i) => {
                        acc_int += i;
                        acc_float += *i as f64;
                    }
                    Val::Float(f) => {
                        is_float = true;
                        acc_float += f;
                    }
                    _ => return Err(format!("+ 不支持类型: {}", a)),
                }
            }
            if is_float {
                Ok(Val::Float(acc_float + acc_int as f64))
            } else {
                Ok(Val::Int(acc_int))
            }
        }
        "-" => {
            if args.is_empty() {
                return Err("- 需要至少一个参数".into());
            }
            let mut acc_int: i64 = 0;
            let mut acc_float: f64 = 0.0;
            let mut is_float = false;
            for (i, a) in args.iter().enumerate() {
                match a {
                    Val::Int(v) => {
                        if i == 0 {
                            acc_int = *v;
                            acc_float = *v as f64;
                        } else {
                            acc_int -= v;
                            acc_float -= *v as f64;
                        }
                    }
                    Val::Float(v) => {
                        is_float = true;
                        if i == 0 {
                            acc_float = *v;
                            acc_int = 0;
                        } else {
                            acc_float -= v;
                        }
                    }
                    _ => return Err(format!("- 不支持类型: {}", a)),
                }
            }
            if is_float {
                Ok(Val::Float(acc_float))
            } else {
                Ok(Val::Int(acc_int))
            }
        }
        "*" => {
            let mut acc_int: i64 = 1;
            let mut acc_float: f64 = 1.0;
            let mut is_float = false;
            for a in args {
                match a {
                    Val::Int(i) => {
                        acc_int *= i;
                        acc_float *= *i as f64;
                    }
                    Val::Float(f) => {
                        is_float = true;
                        acc_float *= f;
                    }
                    _ => return Err(format!("* 不支持类型: {}", a)),
                }
            }
            if is_float {
                Ok(Val::Float(acc_float))
            } else {
                Ok(Val::Int(acc_int))
            }
        }
        "/" => {
            if args.len() != 2 {
                return Err("/ 需要恰好 2 个参数".into());
            }
            match (&args[0], &args[1]) {
                (Val::Int(a), Val::Int(b)) => {
                    if *b == 0 {
                        return Err("除零".into());
                    }
                    Ok(Val::Int(a / b))
                }
                _ => {
                    let a = to_float(&args[0])?;
                    let b = to_float(&args[1])?;
                    if b == 0.0 {
                        return Err("除零".into());
                    }
                    Ok(Val::Float(a / b))
                }
            }
        }
        "=" => {
            if args.len() != 2 {
                return Err("= 需要恰好 2 个参数".into());
            }
            Ok(Val::Bool(args[0] == args[1]))
        }
        "<" => {
            if args.len() != 2 {
                return Err("< 需要恰好 2 个参数".into());
            }
            let a = to_float(&args[0])?;
            let b = to_float(&args[1])?;
            Ok(Val::Bool(a < b))
        }
        _ => Err(format!("未知函数: {}", name)),
    }
}

fn to_float(v: &Val) -> Result<f64, String> {
    match v {
        Val::Int(i) => Ok(*i as f64),
        Val::Float(f) => Ok(*f),
        _ => Err(format!("期望数值: {}", v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn evals_arithmetic() {
        let v = parse("(+ 1 2 3)").unwrap();
        let mut interp = Interp::new();
        let r = interp.eval(&v).unwrap();
        assert!(matches!(r, Val::Int(6)));
    }

    #[test]
    fn evals_if() {
        let v = parse("(if (< 1 2) 10 20)").unwrap();
        let mut interp = Interp::new();
        let r = interp.eval(&v).unwrap();
        assert!(matches!(r, Val::Int(10)));
    }
}
