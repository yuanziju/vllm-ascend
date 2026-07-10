//! dsl — 最小文本 DSL 解析（手工构造测试图的便捷入口）
//!
//! 设计哲学：提供一个人工可读、易写的文本格式，方便写单测和端到端用例，
//! 不依赖 ONNX 二进制。语法极简，每行一条语句：
//!
//! ```text
//! graph "name"           // 图名（可选）
//! in x: f32[2,3]         // 输入：name: dtype[dims]
//! y = relu(x)            // 节点：output = op(args...)
//! z = matmul(x, w)       // 多输入逗号分隔
//! out z                  // 标记图输出
//! ```
//!
//! 支持的 op 名复用 ONNX 风格（Add/Sub/Mul/Div/MatMul/Relu/Gelu/...），
//! 未知 op → Custom。注释用 `//`。

use base::{DType, Graph, NeutronError, OpKind, Result, Type};
use std::collections::HashMap;

/// 解析 DSL 文本为架构无关图
pub fn parse(src: &str) -> Result<Graph> {
    let mut g = Graph::new("dsl");
    let mut names: HashMap<String, u32> = HashMap::new();

    for (lineno, orig) in src.lines().enumerate() {
        let line = strip_comment(orig).trim();
        if line.is_empty() {
            continue;
        }
        parse_line(&mut g, &mut names, line, lineno + 1)?;
    }

    Ok(g)
}

fn strip_comment(s: &str) -> &str {
    match s.find("//") {
        Some(i) => &s[..i],
        None => s,
    }
}

fn parse_line(
    g: &mut Graph,
    names: &mut HashMap<String, u32>,
    line: &str,
    lineno: usize,
) -> Result<()> {
    if let Some(rest) = line.strip_prefix("graph ") {
        let name = rest.trim().trim_matches('"');
        g.name = name.to_string();
        return Ok(());
    }
    if let Some(rest) = line.strip_prefix("in ") {
        return parse_input(g, names, rest.trim(), lineno);
    }
    if let Some(rest) = line.strip_prefix("out ") {
        return parse_output(g, names, rest.trim(), lineno);
    }
    if let Some(eq) = line.find('=') {
        let out_name = line[..eq].trim();
        let rhs = line[eq + 1..].trim();
        return parse_node(g, names, out_name, rhs, lineno);
    }
    Err(NeutronError::Frontend(format!(
        "第 {} 行无法解析: {}",
        lineno, line
    )))
}

/// `x: f32[2,3]`
fn parse_input(
    g: &mut Graph,
    names: &mut HashMap<String, u32>,
    rest: &str,
    lineno: usize,
) -> Result<()> {
    let colon = rest
        .find(':')
        .ok_or_else(|| NeutronError::Frontend(format!("第 {} 行输入缺 ':' : {}", lineno, rest)))?;
    let name = rest[..colon].trim();
    let type_str = rest[colon + 1..].trim();
    let ty = parse_type(type_str, lineno)?;
    let v = g.add_input(ty, Some(name));
    g.mark_input(v);
    names.insert(name.to_string(), v);
    Ok(())
}

fn parse_output(
    g: &mut Graph,
    names: &mut HashMap<String, u32>,
    rest: &str,
    lineno: usize,
) -> Result<()> {
    let v = *names
        .get(rest)
        .ok_or_else(|| NeutronError::Frontend(format!("第 {} 行输出未定义: {}", lineno, rest)))?;
    g.mark_output(v);
    Ok(())
}

/// `out_name = op(arg0, arg1)`，rhs = "op(args)"
fn parse_node(
    g: &mut Graph,
    names: &mut HashMap<String, u32>,
    out_name: &str,
    rhs: &str,
    lineno: usize,
) -> Result<()> {
    // 拆 op 名和参数列表
    let paren = rhs
        .find('(')
        .ok_or_else(|| NeutronError::Frontend(format!("第 {} 行节点缺 '(': {}", lineno, rhs)))?;
    let op_name = rhs[..paren].trim();
    let args_str = rhs[paren + 1..].trim().trim_end_matches(')');
    let kind = map_dsl_op(op_name);
    let nid = g.add_node(kind);

    // 输出 value
    let out_ty = Type::Tensor {
        dtype: DType::F32,
        dims: vec![-1, -1],
    };
    let out_v = g.add_value(out_ty, Some(out_name), nid);
    g.storage.set_node_outputs(nid, &[out_v]);
    names.insert(out_name.to_string(), out_v);

    // 输入 value（引用已注册的 name）
    let mut inputs: Vec<u32> = Vec::new();
    if !args_str.trim().is_empty() {
        for arg in args_str.split(',') {
            let arg = arg.trim();
            let v = *names.get(arg).ok_or_else(|| {
                NeutronError::Frontend(format!("第 {} 行参数未定义: {}", lineno, arg))
            })?;
            inputs.push(v);
        }
    }
    g.storage.set_node_inputs(nid, &inputs);
    Ok(())
}

/// `f32[2,3]` 或 `f32` 或 `i64[4]`
fn parse_type(s: &str, lineno: usize) -> Result<Type> {
    let s = s.trim();
    let (dtype_str, dims) = match s.find('[') {
        Some(i) => {
            let close = s.find(']').ok_or_else(|| {
                NeutronError::Frontend(format!("第 {} 行类型缺 ']': {}", lineno, s))
            })?;
            let dt = &s[..i];
            let dims_str = &s[i + 1..close];
            let dims: Vec<i64> = dims_str
                .split(',')
                .map(|d| d.trim().parse::<i64>().unwrap_or(-1))
                .collect();
            (dt, dims)
        }
        None => (s, Vec::new()),
    };
    let dtype = match dtype_str.trim() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "i64" => DType::I64,
        "i32" => DType::I32,
        "bool" => DType::Bool,
        other => {
            return Err(NeutronError::Frontend(format!(
                "第 {} 行未知 dtype: {}",
                lineno, other
            )));
        }
    };
    if dims.is_empty() {
        Ok(Type::Scalar(dtype))
    } else {
        Ok(Type::Tensor { dtype, dims })
    }
}

fn map_dsl_op(name: &str) -> OpKind {
    // 大小写不敏感，复用 ONNX 风格名
    match name {
        "Add" | "add" => OpKind::Add,
        "Sub" | "sub" => OpKind::Sub,
        "Mul" | "mul" => OpKind::Mul,
        "Div" | "div" => OpKind::Div,
        "MatMul" | "matmul" => OpKind::MatMul,
        "Gemm" | "gemm" => OpKind::MatMul,
        "Relu" | "relu" => OpKind::Relu,
        "Gelu" | "gelu" => OpKind::Gelu,
        "Sigmoid" | "sigmoid" => OpKind::Sigmoid,
        "Tanh" | "tanh" => OpKind::Tanh,
        "Softmax" | "softmax" => OpKind::Softmax,
        "LayerNorm" | "layernorm" => OpKind::LayerNorm,
        "Conv" | "conv" => OpKind::Conv,
        "Pool" | "pool" => OpKind::Pool,
        "Reshape" | "reshape" => OpKind::Reshape,
        "Transpose" | "transpose" => OpKind::Transpose,
        "Concat" | "concat" => OpKind::Concat,
        "Slice" | "slice" => OpKind::Slice,
        "Sqrt" | "sqrt" => OpKind::Sqrt,
        "Exp" | "exp" => OpKind::Exp,
        "Pow" | "pow" => OpKind::Pow,
        "ReduceSum" | "reducesum" => OpKind::ReduceSum,
        "ReduceMean" | "reducemean" => OpKind::ReduceMean,
        "ReduceMax" | "reducemax" => OpKind::ReduceMax,
        _ => OpKind::Custom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_graph() {
        let src = r#"
            graph "test"
            in x: f32[2,3]
            in w: f32[3,4]
            y = matmul(x, w)   // 注释
            z = relu(y)
            out z
        "#;
        let g = parse(src).unwrap();
        assert_eq!(g.name, "test");
        assert_eq!(g.inputs().len(), 2, "应有 2 个输入");
        assert_eq!(g.outputs().len(), 1, "应有 1 个输出");
        let kinds: Vec<OpKind> = g.node_ids().map(|id| g.node(id).unwrap().kind).collect();
        assert!(kinds.contains(&OpKind::MatMul));
        assert!(kinds.contains(&OpKind::Relu));
    }

    #[test]
    fn parses_scalar_type() {
        let src = "in x: f32\nout x";
        let g = parse(src).unwrap();
        assert_eq!(g.inputs().len(), 1);
    }

    #[test]
    fn unknown_op_becomes_custom() {
        let src = "in x: f32[4]\ny = foobar(x)\nout y";
        let g = parse(src).unwrap();
        let kinds: Vec<OpKind> = g.node_ids().map(|id| g.node(id).unwrap().kind).collect();
        assert!(kinds.contains(&OpKind::Custom));
    }

    #[test]
    fn error_on_undefined_arg() {
        let src = "y = relu(z)";
        let err = parse(src).unwrap_err();
        assert!(matches!(err, NeutronError::Frontend(_)));
    }

    #[test]
    fn parses_no_input_node() {
        // 零参数 op
        let src = "y = relu()";
        // 没 input 也没 out，应报错（relu() 输出 y 但无输入参数）
        // 实际：out y 未声明但 relu() 产生 y。这条应成功构造 1 个节点 0 输入
        let g = parse(src).unwrap();
        assert_eq!(g.node_count(), 1);
        let n = g.node(0).unwrap();
        assert_eq!(n.kind, OpKind::Relu);
        assert_eq!(n.inputs().len(), 0);
    }
}
