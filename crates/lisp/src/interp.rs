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
            Val::Nil
            | Val::Bool(_)
            | Val::Int(_)
            | Val::Float(_)
            | Val::Str(_)
            | Val::Lambda { .. } => Ok(v.clone()),
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
                            items.get(3).map(|v| self.eval(v)).unwrap_or(Ok(Val::Nil))
                        }
                    }
                    "do" => {
                        let mut result = Val::Nil;
                        for item in &items[1..] {
                            result = self.eval(item)?;
                        }
                        Ok(result)
                    }
                    // 短路逻辑 and / or（特殊形式，不求值全部参数）
                    "and" => {
                        let mut result = Val::Bool(true);
                        for item in &items[1..] {
                            result = self.eval(item)?;
                            if matches!(result, Val::Bool(false) | Val::Nil) {
                                return Ok(result);
                            }
                        }
                        Ok(result)
                    }
                    "or" => {
                        for item in &items[1..] {
                            let v = self.eval(item)?;
                            if !matches!(v, Val::Bool(false) | Val::Nil) {
                                return Ok(v);
                            }
                        }
                        Ok(Val::Bool(false))
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
            // 注意：is_float=true 时 acc_float 已含所有 Int 累加，不能再 + acc_int
            // （早期 bug：重复加 Int 导致 (+ 1 2.5)=4.5 而非 3.5）
            if is_float {
                Ok(Val::Float(acc_float))
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
            // 同 +：is_float=true 时 acc_float 已含所有 Int 累乘
            // （早期 bug：返回 acc_float + acc_int as f64，类型错且语义错）
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
        "not" => {
            if args.len() != 1 {
                return Err("not 需要恰好 1 个参数".into());
            }
            Ok(Val::Bool(matches!(&args[0], Val::Bool(false) | Val::Nil)))
        }
        // 字符串拼接：所有参数转字符串后拼接
        "str" => {
            let mut s = String::new();
            for a in args {
                s.push_str(&val_to_str(a));
            }
            Ok(Val::Str(s))
        }
        // 字符串相等（= 已用 PartialEq 处理，但 Sym!=Str 时补一个显式 str=）
        "str=" => {
            if args.len() != 2 {
                return Err("str= 需要恰好 2 个参数".into());
            }
            Ok(Val::Bool(val_to_str(&args[0]) == val_to_str(&args[1])))
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

/// 把 lisp Val 转成字符串（str/str= 内建函数用）
fn val_to_str(v: &Val) -> String {
    match v {
        Val::Nil => "nil".to_string(),
        Val::Bool(b) => b.to_string(),
        Val::Int(i) => i.to_string(),
        Val::Float(f) => f.to_string(),
        Val::Sym(s) => s.clone(),
        Val::Str(s) => s.clone(),
        Val::List(items) => {
            let mut s = String::from("(");
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                s.push_str(&val_to_str(it));
            }
            s.push(')');
            s
        }
        Val::Lambda { .. } => "<lambda>".to_string(),
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

    fn eval_str(src: &str) -> Val {
        let v = parse(src).unwrap();
        let mut interp = Interp::new();
        interp.eval(&v).unwrap()
    }

    fn eval_err(src: &str) -> String {
        let v = parse(src).unwrap();
        let mut interp = Interp::new();
        interp.eval(&v).unwrap_err()
    }

    #[test]
    fn sub_and_mul_work() {
        assert!(matches!(eval_str("(- 10 3 2)"), Val::Int(5)));
        assert!(matches!(eval_str("(* 2 3 4)"), Val::Int(24)));
        // 单参数 -
        assert!(matches!(eval_str("(- 5)"), Val::Int(5)));
        // 单参数 *
        assert!(matches!(eval_str("(* 7)"), Val::Int(7)));
    }

    #[test]
    fn mixed_int_float_promotes_to_float() {
        // int + float → float
        match eval_str("(+ 1 2.5)") {
            Val::Float(f) => assert!((f - 3.5).abs() < 1e-12),
            other => panic!("期望 Float，得到 {:?}", other),
        }
        match eval_str("(- 10.0 3)") {
            Val::Float(f) => assert!((f - 7.0).abs() < 1e-12),
            other => panic!("期望 Float，得到 {:?}", other),
        }
        match eval_str("(* 2.0 3)") {
            Val::Float(f) => assert!((f - 6.0).abs() < 1e-12),
            _ => panic!("期望 Float"),
        }
    }

    #[test]
    fn division_int_and_float() {
        assert!(matches!(eval_str("(/ 10 2)"), Val::Int(5)));
        match eval_str("(/ 10.0 3)") {
            Val::Float(f) => assert!((f - 10.0 / 3.0).abs() < 1e-12),
            _ => panic!("期望 Float"),
        }
        // 整除向零取整
        assert!(matches!(eval_str("(/ -7 2)"), Val::Int(-3)));
    }

    #[test]
    fn division_by_zero_errors() {
        assert!(eval_err("(/ 5 0)").contains("除零"));
        assert!(eval_err("(/ 5.0 0.0)").contains("除零"));
    }

    #[test]
    fn short_circuit_and_or() {
        // and 短路：第一个假就返回，不评估后续
        assert!(matches!(eval_str("(and true true)"), Val::Bool(true)));
        assert!(matches!(eval_str("(and true false)"), Val::Bool(false)));
        assert!(matches!(
            eval_str("(and false undefined-sym)"),
            Val::Bool(false)
        ));

        // or 短路：第一个真就返回
        assert!(matches!(eval_str("(or false 42)"), Val::Int(42)));
        assert!(matches!(eval_str("(or false false)"), Val::Bool(false)));
        // or 第一个真 → 不评估后续
        assert!(matches!(eval_str("(or 42 undefined-sym)"), Val::Int(42)));
    }

    #[test]
    fn do_block_returns_last() {
        assert!(matches!(eval_str("(do 1 2 3)"), Val::Int(3)));
        // 空 do 返回 nil
        assert!(matches!(eval_str("(do)"), Val::Nil));
    }

    #[test]
    fn quote_returns_unevaled() {
        // (quote sym) 返回 sym 本身（不求值）
        match eval_str("(quote sym)") {
            Val::Sym(s) => assert_eq!(s, "sym"),
            other => panic!("期望 Sym，得到 {:?}", other),
        }
        // (quote (1 2 3)) 返回 list
        match eval_str("(quote (1 2 3))") {
            Val::List(items) => assert_eq!(items.len(), 3),
            _ => panic!("期望 List"),
        }
    }

    #[test]
    fn if_without_else_returns_nil_when_false() {
        assert!(matches!(eval_str("(if false 42)"), Val::Nil));
    }

    #[test]
    fn not_builtin() {
        assert!(matches!(eval_str("(not false)"), Val::Bool(true)));
        assert!(matches!(eval_str("(not nil)"), Val::Bool(true)));
        assert!(matches!(eval_str("(not 42)"), Val::Bool(false)));
    }

    #[test]
    fn equality_and_inequality() {
        assert!(matches!(eval_str("(= 1 1)"), Val::Bool(true)));
        assert!(matches!(eval_str("(= 1 2)"), Val::Bool(false)));
        assert!(matches!(eval_str("(< 1 2)"), Val::Bool(true)));
        assert!(matches!(eval_str("(< 2 1)"), Val::Bool(false)));
        // 字符串相等
        assert!(matches!(eval_str(r#"(str= "foo" "foo")"#), Val::Bool(true)));
        assert!(matches!(
            eval_str(r#"(str= "foo" "bar")"#),
            Val::Bool(false)
        ));
    }

    #[test]
    fn str_concatenation() {
        match eval_str(r#"(str "hello" " " "world")"#) {
            Val::Str(s) => assert_eq!(s, "hello world"),
            _ => panic!("期望 Str"),
        }
    }

    #[test]
    fn unbound_symbol_errors() {
        let err = eval_err("undefined_sym");
        assert!(err.contains("未绑定符号"));
    }

    #[test]
    fn unknown_function_errors() {
        let err = eval_err("(unknown_fn 1 2)");
        assert!(err.contains("未知函数"));
    }

    #[test]
    fn wrong_arity_errors() {
        // / 需要恰好 2 个参数
        assert!(eval_err("(/ 1 2 3)").contains("2 个参数"));
        // < 需要恰好 2 个参数
        assert!(eval_err("(< 1 2 3)").contains("2 个参数"));
        // not 需要恰好 1 个参数
        assert!(eval_err("(not true false)").contains("1 个参数"));
    }

    #[test]
    fn nil_evaluates_to_nil() {
        assert!(matches!(eval_str("nil"), Val::Nil));
        assert!(matches!(eval_str("true"), Val::Bool(true)));
        assert!(matches!(eval_str("false"), Val::Bool(false)));
    }

    #[test]
    fn empty_list_evaluates_to_empty_list() {
        assert!(matches!(eval_str("()"), Val::List(_)));
    }
}
