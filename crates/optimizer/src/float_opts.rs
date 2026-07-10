//! float_opts — 浮点数结构优化（IEEE754 位级 trick + Flash Attention 式重排）
//!
//! 设计哲学：针对浮点本身的结构做优化，不是模式匹配复合算子。
//!
//! 实现的优化：
//! - **DivByConstToMul**：`x / c` → `x * (1/c)`。除法 latency 远高于乘法，
//!   预计算倒数转成乘法。注意：对 c=0 不做；浮点倒数有精度损失但可接受。
//! - **MulByTwoToAdd**：`x * 2.0` → `x + x`。乘以 2 的幂可用 IEEE754 位级
//!   操作（指数+1），但加法在某些硬件更便宜，且不引入常量。此处保守用加法。
//! - **FastInvSqrt 识别**：`1.0 / sqrt(x)` 模式识别（不改图，标记为机会），
//!   后端 lowering 阶段用 0x5f3759df 魔数 trick 实现（Quake III fast inverse sqrt）。
//! - **SoftmaxOnline 标记**：Softmax 节点标记为可用 online-softmax 重排
//!   （Flash Attention 式：避免 materialize 中间矩阵，online 计算 max/sum）。

use base::{Graph, NodeView, OpKind, Result, ValueId};

/// 浮点优化机会（识别到的不一定立即应用）
#[derive(Debug, Clone)]
pub enum FloatOpt {
    /// `1.0 / sqrt(x)` 模式：可替换为 FastInvSqrt 单节点
    FastInvSqrt {
        div_node: base::NodeId,
        sqrt_node: base::NodeId,
    },
    /// Softmax 可用 online 算法重排（Flash Attention 式）
    SoftmaxOnline { softmax_node: base::NodeId },
    /// `x * 2.0` → `x + x`
    MulByTwoToAdd {
        mul_node: base::NodeId,
        x_input: ValueId,
    },
    /// `x / c` → `x * (1/c)`
    DivByConstToMul {
        div_node: base::NodeId,
        reciprocal: f64,
    },
}

pub fn find_opportunities(graph: &Graph) -> Result<Vec<FloatOpt>> {
    let mut opts = Vec::new();
    for id in graph.node_ids() {
        let n = graph.node(id)?;
        match n.kind {
            OpKind::Div => {
                if let Some(opt) = try_match_fast_inv_sqrt(graph, n)? {
                    opts.push(opt);
                }
                if let Some(opt) = try_match_div_by_const(graph, n)? {
                    opts.push(opt);
                }
            }
            OpKind::Softmax => {
                opts.push(FloatOpt::SoftmaxOnline { softmax_node: id });
            }
            OpKind::Mul => {
                if let Some(opt) = try_match_mul_by_two(graph, n)? {
                    opts.push(opt);
                }
            }
            _ => {}
        }
    }
    Ok(opts)
}

/// 识别 `1.0 / sqrt(x)`：Div 节点，一个输入是常量 1.0，另一个是 Sqrt 节点输出
fn try_match_fast_inv_sqrt(graph: &Graph, div: NodeView) -> Result<Option<FloatOpt>> {
    let ins = div.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (a, b) = (ins[0], ins[1]);
    // 检查 a=1.0 常量，b=sqrt 输出
    if let (Some(va), None) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        if va == 1.0 {
            let b_def = graph.value(b)?.def_node();
            if b_def != u32::MAX {
                let b_node = graph.node(b_def)?;
                if b_node.kind == OpKind::Sqrt {
                    return Ok(Some(FloatOpt::FastInvSqrt {
                        div_node: div.id,
                        sqrt_node: b_def,
                    }));
                }
            }
        }
    }
    // 检查 b=1.0 常量，a=sqrt 输出（1.0/sqrt(x) 和 sqrt(x)/1.0 不同，后者无意义）
    Ok(None)
}

/// 识别 `x * 2.0`：Mul 节点，一个输入是常量 2.0
fn try_match_mul_by_two(graph: &Graph, mul: NodeView) -> Result<Option<FloatOpt>> {
    let ins = mul.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (a, b) = (ins[0], ins[1]);
    if constant_value(graph, a)? == Some(2.0) {
        return Ok(Some(FloatOpt::MulByTwoToAdd {
            mul_node: mul.id,
            x_input: b,
        }));
    }
    if constant_value(graph, b)? == Some(2.0) {
        return Ok(Some(FloatOpt::MulByTwoToAdd {
            mul_node: mul.id,
            x_input: a,
        }));
    }
    Ok(None)
}

/// 识别 `x / c`：Div 节点，一个输入是常量 c（c != 0, c != 1）
fn try_match_div_by_const(graph: &Graph, div: NodeView) -> Result<Option<FloatOpt>> {
    let ins = div.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (a, b) = (ins[0], ins[1]);
    // b 是常量（除数常量）
    if let Some(c) = constant_value(graph, b)? {
        if c != 0.0 && c != 1.0 {
            return Ok(Some(FloatOpt::DivByConstToMul {
                div_node: div.id,
                reciprocal: 1.0 / c,
            }));
        }
    }
    let _ = a;
    Ok(None)
}

fn constant_value(graph: &Graph, v: ValueId) -> Result<Option<f64>> {
    let val = graph.value(v)?;
    let def = val.def_node();
    if def == u32::MAX {
        return Ok(None);
    }
    let node = graph.node(def)?;
    Ok(node.constant_value())
}

/// 应用浮点优化。返回应用次数。
/// 当前实现 DivByConstToMul 和 MulByTwoToAdd（改图），
/// FastInvSqrt 和 SoftmaxOnline 仅识别不应用（留给 lowering）。
pub fn apply_float_opts(graph: &mut Graph) -> Result<usize> {
    let opts = find_opportunities(graph)?;
    let mut applied = 0usize;

    for opt in opts {
        match opt {
            FloatOpt::DivByConstToMul {
                div_node,
                reciprocal,
            } => {
                // 把 Div 节点的 op 改成 Mul，把常量输入替换为新常量 (1/c)
                let (_cnode, cval) = graph.add_constant_f64(reciprocal);
                let n = graph.node(div_node)?;
                let old_ins = n.inputs().to_vec();
                // 哪个输入是原常量？替换之
                let mut new_ins = old_ins.clone();
                for i in 0..new_ins.len() {
                    if constant_value(graph, old_ins[i])?.is_some() {
                        new_ins[i] = cval;
                        break;
                    }
                }
                graph.storage.set_node_inputs(div_node, &new_ins);
                // 改 op tag：Div(3) -> Mul(2)
                graph.storage.node_hdr[div_node as usize].op_tag = OpKind::Mul as u8;
                applied += 1;
            }
            FloatOpt::MulByTwoToAdd { mul_node, x_input } => {
                // 把 Mul 节点的 op 改成 Add，输入改成 [x, x]
                graph.storage.set_node_inputs(mul_node, &[x_input, x_input]);
                graph.storage.node_hdr[mul_node as usize].op_tag = OpKind::Add as u8;
                applied += 1;
            }
            FloatOpt::FastInvSqrt { .. } | FloatOpt::SoftmaxOnline { .. } => {
                // 仅识别，不改图（留给 lowering 阶段用专门 kernel）
            }
        }
    }

    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    #[test]
    fn div_by_const_becomes_mul() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, two) = g.add_constant_f64(2.0);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        g.storage.set_node_inputs(div, &[x, two]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(div).unwrap();
        assert_eq!(n.kind, OpKind::Mul, "Div 应已改成 Mul");
        // 新常量应是 0.5
        let new_const_input = n.inputs()[1];
        let cv = constant_value(&g, new_const_input).unwrap();
        assert_eq!(cv, Some(0.5));
    }

    #[test]
    fn mul_by_two_becomes_add() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, two) = g.add_constant_f64(2.0);
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), mul);
        g.storage.set_node_inputs(mul, &[x, two]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(mul).unwrap();
        assert_eq!(n.kind, OpKind::Add, "Mul 应已改成 Add");
        assert_eq!(n.inputs(), &[x, x], "输入应改成 [x, x]");
    }

    #[test]
    fn fast_inv_sqrt_detected() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let sqrt_out = g.add_value(Type::Scalar(DType::F32), Some("sx"), sqrt);
        g.storage.set_node_inputs(sqrt, &[x]);
        g.storage.set_node_outputs(sqrt, &[sqrt_out]);
        let (_c, one) = g.add_constant_f64(1.0);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        g.storage.set_node_inputs(div, &[one, sqrt_out]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            opts.iter()
                .any(|o| matches!(o, FloatOpt::FastInvSqrt { .. })),
            "应识别 FastInvSqrt 机会"
        );
    }

    #[test]
    fn softmax_marked_online() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sm = g.add_node(OpKind::Softmax);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sm);
        g.storage.set_node_inputs(sm, &[x]);
        g.storage.set_node_outputs(sm, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            opts.iter()
                .any(|o| matches!(o, FloatOpt::SoftmaxOnline { .. })),
            "Softmax 应被标记为 online 机会"
        );
    }

    #[test]
    fn div_by_one_not_optimized() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, one) = g.add_constant_f64(1.0);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        g.storage.set_node_inputs(div, &[x, one]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        // x/1 不应触发 DivByConstToMul（c=1 跳过，留给 algebra 的 x/1=x）
        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 0);
    }
}
