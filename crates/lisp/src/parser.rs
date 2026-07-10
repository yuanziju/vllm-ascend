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
}
