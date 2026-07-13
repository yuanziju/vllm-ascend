//! parser — S-expr 文本解析器

use crate::Val;
use std::iter::Peekable;
use std::str::Chars;

/// 解析 S-expr 文本
pub fn parse(src: &str) -> Result<Val, String> {
    let mut p = Parser {
        chars: src.chars().peekable(),
    };
    p.skip_ws();
    if p.chars.peek().is_none() {
        return Ok(Val::Nil);
    }
    let v = p.parse_expr()?;
    p.skip_ws();
    Ok(v)
}

struct Parser<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
            } else if c == ';' {
                // 行注释
                for c in self.chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Val, String> {
        self.skip_ws();
        match self.chars.peek() {
            None => Err("意外的输入结束".into()),
            Some(&'(') => self.parse_list(),
            Some(&'"') => self.parse_string(),
            Some(&'\'') => {
                self.chars.next();
                Ok(Val::Str("quote".into()))
            }
            Some(_) => self.parse_atom(),
        }
    }

    /// 解析双引号字符串字面量（支持 \" 转义）
    fn parse_string(&mut self) -> Result<Val, String> {
        // 吃掉开头的 '"'
        self.chars.next();
        let mut s = String::new();
        loop {
            match self.chars.next() {
                None => return Err("未闭合的字符串字面量".into()),
                Some('"') => return Ok(Val::Str(s)),
                Some('\\') => match self.chars.next() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some(c) => s.push(c),
                    None => return Err("转义序列意外结束".into()),
                },
                Some(c) => s.push(c),
            }
        }
    }

    fn parse_list(&mut self) -> Result<Val, String> {
        // 吃掉 '('
        self.chars.next();
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.chars.peek() {
                None => return Err("未闭合的 '('".into()),
                Some(&')') => {
                    self.chars.next();
                    return Ok(Val::List(items));
                }
                Some(_) => {
                    items.push(self.parse_expr()?);
                }
            }
        }
    }

    fn parse_atom(&mut self) -> Result<Val, String> {
        let mut s = String::new();
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() || c == '(' || c == ')' {
                break;
            }
            s.push(c);
            self.chars.next();
        }
        Ok(atom_to_val(&s))
    }
}

fn atom_to_val(s: &str) -> Val {
    if s == "nil" {
        return Val::Nil;
    }
    if s == "true" {
        return Val::Bool(true);
    }
    if s == "false" {
        return Val::Bool(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return Val::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return Val::Float(f);
    }
    Val::Sym(s.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_atoms() {
        assert!(matches!(parse("42").unwrap(), Val::Int(42)));
        assert!(matches!(parse("3.14").unwrap(), Val::Float(_)));
        assert!(matches!(parse("foo").unwrap(), Val::Sym(_)));
    }

    #[test]
    fn parses_list() {
        let v = parse("(add 1 2)").unwrap();
        let items = v.as_list().unwrap();
        assert_eq!(items.len(), 3);
    }

    /// nil / true / false 字面量
    #[test]
    fn parses_bool_literals() {
        assert!(matches!(parse("nil").unwrap(), Val::Nil));
        assert!(matches!(parse("true").unwrap(), Val::Bool(true)));
        assert!(matches!(parse("false").unwrap(), Val::Bool(false)));
    }

    /// 字符串字面量 + 转义序列
    #[test]
    fn parses_string_with_escapes() {
        let v = parse(r#""hello\nworld""#).unwrap();
        match v {
            Val::Str(s) => assert_eq!(s, "hello\nworld"),
            other => panic!("期望 Str，得到 {:?}", other),
        }

        let v = parse(r#""tab\there""#).unwrap();
        match v {
            Val::Str(s) => assert_eq!(s, "tab\there"),
            _ => panic!("期望 Str"),
        }

        // \" 转义
        let v = parse(r#""quote \"inside\"""#).unwrap();
        match v {
            Val::Str(s) => assert_eq!(s, "quote \"inside\""),
            _ => panic!("期望 Str"),
        }
    }

    /// 嵌套 list
    #[test]
    fn parses_nested_lists() {
        let v = parse("(+ 1 (+ 2 3) 4)").unwrap();
        let outer = v.as_list().unwrap();
        assert_eq!(outer.len(), 4);
        let inner = outer[2].as_list().unwrap();
        assert_eq!(inner.len(), 3);
    }

    /// 空列表
    #[test]
    fn parses_empty_list() {
        let v = parse("()").unwrap();
        let items = v.as_list().unwrap();
        assert!(items.is_empty());
    }

    /// 行注释（; 到行尾）
    #[test]
    fn parses_with_comments() {
        let v = parse("; 这是注释\n(+ 1 2) ; 行尾注释").unwrap();
        let items = v.as_list().unwrap();
        assert_eq!(items.len(), 3);
    }

    /// quote 简写：'x 等价于 (quote x)
    #[test]
    fn parses_quote_shorthand() {
        let v = parse("'foo").unwrap();
        match v {
            Val::Str(s) => assert_eq!(s, "quote"),
            _ => panic!("期望 Str(\"quote\")"),
        }
    }

    /// 空输入应返回 Nil
    #[test]
    fn parses_empty_input_as_nil() {
        let v = parse("").unwrap();
        assert!(matches!(v, Val::Nil));

        let v = parse("   ").unwrap();
        assert!(matches!(v, Val::Nil));

        let v = parse("; just a comment").unwrap();
        assert!(matches!(v, Val::Nil));
    }

    /// 未闭合的括号应返回错误
    #[test]
    fn parse_unclosed_paren_errors() {
        assert!(parse("(+ 1 2").is_err());
        assert!(parse("((()").is_err());
    }

    /// 未闭合的字符串字面量应返回错误
    #[test]
    fn parse_unclosed_string_errors() {
        assert!(parse(r#""unclosed"#).is_err());
    }
}
