//! float_opts — 浮点数结构优化（IEEE754 位级 trick + Flash Attention 式重排）
//!
//! 设计哲学：针对浮点本身的结构做优化，不是模式匹配复合算子。
//!
//! 实现的优化：
//! - **FastInvSqrt 融合**：`a / sqrt(b)` → `Mul(a, Rsqrt(b))`（a==1.0 时直接 → `Rsqrt(b)`，
//!   2 op 降 1 op）。恒等式 `a/√b = a·b^(-1/2)`，把 Sqrt+Div（含一个贵的 Div）融成
//!   Rsqrt（单条硬件指令 / 0x5f3759df 魔数 bit trick，Quake III fast inverse sqrt）+ 便宜的 Mul。
//!   RMSNorm/LayerNorm 等 normalization 到处出现。这是浮点结构优化，不是贪心模式匹配。
//! - **ReciprocalSqrt 融合**：`Reciprocal(Sqrt(x))` → `Rsqrt(x)`（2 op 降 1 op）。
//!   同 `1/√x = x^(-1/2)` 恒等式。ONNX 的 Reciprocal(Sqrt(...)) 模式（RMSNorm 常见）
//!   原本需两 op，融成单 Rsqrt。
//! - **DivByReciprocal 融合**：`a / Reciprocal(b)` → `Mul(a, b)`（消去 Reciprocal+Div，
//!   换便宜 Mul）。恒等式 `a/(1/b) = a·b`。除以倒数等于乘原数。
//! - **DivByConstToMul**：`x / c` → `x * (1/c)`。除法 latency 远高于乘法，
//!   预计算倒数转成乘法。注意：对 c=0 不做；浮点倒数有精度损失但可接受。
//! - **MulByTwoToAdd**：`x * 2.0` → `x + x`。乘以 2 的幂可用 IEEE754 位级
//!   操作（指数+1），但加法在某些硬件更便宜，且不引入常量。此处保守用加法。
//! - **SoftmaxOnline 标记**：仅识别不改图。真正的 Flash Attention 融合（softmax+matmul）
//!   是设计哲学禁止的贪心模式匹配；online-softmax 本质是 kernel tiling 策略非 IR 重写，
//!   留作 lowering 阶段的 kernel 机会标记。

use base::{Graph, NodeView, OpKind, Result, ValueId};

/// 浮点优化机会（识别到的不一定立即应用）
#[derive(Debug, Clone)]
pub enum FloatOpt {
    /// `a / sqrt(b)` 模式：融合为 Rsqrt。恒等式 `a/√b = a·b^(-1/2)`。
    /// a==1.0 常量时直接 → `Rsqrt(b)`（2 op 降 1 op）；否则 → `Mul(a, Rsqrt(b))`
    FastInvSqrt {
        div_node: base::NodeId,
        sqrt_node: base::NodeId,
        numerator: ValueId,
        sqrt_input: ValueId,
        numerator_is_one: bool,
    },
    /// Softmax 可用 online 算法重排（Flash Attention 式，仅标记不改图）
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
    /// `Reciprocal(Sqrt(x))` → `Rsqrt(x)`（2 op 降 1 op）。同 1/√x = x^(-1/2) 恒等式
    ReciprocalSqrt {
        recip_node: base::NodeId,
        sqrt_node: base::NodeId,
        sqrt_input: ValueId,
    },
    /// `a / Reciprocal(b)` → `Mul(a, b)`（消去 Reciprocal+Div 换便宜 Mul）。a/(1/b)=a·b
    DivByReciprocal {
        div_node: base::NodeId,
        numerator: ValueId,
        recip_input: ValueId,
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
                if let Some(opt) = try_match_div_by_reciprocal(graph, n)? {
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
            OpKind::Reciprocal => {
                if let Some(opt) = try_match_reciprocal_sqrt(graph, n)? {
                    opts.push(opt);
                }
            }
            _ => {}
        }
    }
    Ok(opts)
}

/// 识别 `a / sqrt(b)`：Div 节点，除数(ins[1]) 是 Sqrt 节点输出。
/// 分子 a 可以是任意 value（常量或非常量）。`a/√b = a·b^(-1/2)` 融合为 Rsqrt。
/// a==1.0 常量时直接 → Rsqrt(b)（2 op 降 1 op）；否则 → Mul(a, Rsqrt(b))。
/// 注意：只匹配除数是 Sqrt 的情况（`sqrt(x)/a` ≠ `a·rsqrt(x)`，不匹配）
fn try_match_fast_inv_sqrt(graph: &Graph, div: NodeView) -> Result<Option<FloatOpt>> {
    let ins = div.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (numerator, divisor) = (ins[0], ins[1]);
    // 除数必须是 Sqrt 节点输出
    let divisor_def = graph.value(divisor)?.def_node();
    if divisor_def == u32::MAX {
        return Ok(None);
    }
    let divisor_node = graph.node(divisor_def)?;
    if divisor_node.kind != OpKind::Sqrt {
        return Ok(None);
    }
    let Some(&sqrt_input) = divisor_node.inputs().first() else {
        return Ok(None);
    };
    // 分子是否为常量 1.0（特殊case：直接 → Rsqrt，省一个 Mul）
    let numerator_is_one = matches!(constant_value(graph, numerator)?, Some(v) if v == 1.0);
    Ok(Some(FloatOpt::FastInvSqrt {
        div_node: div.id,
        sqrt_node: divisor_def,
        numerator,
        sqrt_input,
        numerator_is_one,
    }))
}

/// 识别 `Reciprocal(Sqrt(x))`：Reciprocal 节点，输入(ins[0]) 是 Sqrt 节点输出。
/// `1/√x = x^(-1/2)` 融合为 Rsqrt（2 op 降 1 op）。
/// ONNX 的 Reciprocal(Sqrt(...)) 是 RMSNorm 常见模式（比 Div(1,Sqrt) 另一种写法）
fn try_match_reciprocal_sqrt(graph: &Graph, recip: NodeView) -> Result<Option<FloatOpt>> {
    let ins = recip.inputs();
    if ins.len() != 1 {
        return Ok(None);
    }
    let input = ins[0];
    let input_def = graph.value(input)?.def_node();
    if input_def == u32::MAX {
        return Ok(None);
    }
    let input_node = graph.node(input_def)?;
    if input_node.kind != OpKind::Sqrt {
        return Ok(None);
    }
    let Some(&sqrt_input) = input_node.inputs().first() else {
        return Ok(None);
    };
    Ok(Some(FloatOpt::ReciprocalSqrt {
        recip_node: recip.id,
        sqrt_node: input_def,
        sqrt_input,
    }))
}

/// 识别 `a / Reciprocal(b)`：Div 节点，除数(ins[1]) 是 Reciprocal 节点输出。
/// `a/(1/b) = a·b`，消去 Reciprocal+Div 换便宜 Mul。注意：只匹配除数是 Reciprocal
/// （分子是 Reciprocal 不匹配，那是 Reciprocal(a)/b 无此恒等式）
fn try_match_div_by_reciprocal(graph: &Graph, div: NodeView) -> Result<Option<FloatOpt>> {
    let ins = div.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (numerator, divisor) = (ins[0], ins[1]);
    let divisor_def = graph.value(divisor)?.def_node();
    if divisor_def == u32::MAX {
        return Ok(None);
    }
    let divisor_node = graph.node(divisor_def)?;
    if divisor_node.kind != OpKind::Reciprocal {
        return Ok(None);
    }
    let Some(&recip_input) = divisor_node.inputs().first() else {
        return Ok(None);
    };
    Ok(Some(FloatOpt::DivByReciprocal {
        div_node: div.id,
        numerator,
        recip_input,
    }))
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
/// FastInvSqrt / DivByConstToMul / MulByTwoToAdd 改图；SoftmaxOnline 仅标记不改图。
pub fn apply_float_opts(graph: &mut Graph) -> Result<usize> {
    let opts = find_opportunities(graph)?;
    let mut applied = 0usize;

    for opt in opts {
        match opt {
            FloatOpt::FastInvSqrt {
                div_node,
                sqrt_node: _,
                numerator,
                sqrt_input,
                numerator_is_one,
            } => {
                if numerator_is_one {
                    // 1.0 / sqrt(b) → Rsqrt(b)：把 Div 节点本身改成 Rsqrt，输入换成 b。
                    // Div 的输出 value 不变（使用者仍指向它），Sqrt + 1.0 常量变孤儿交给 DCE
                    graph.storage.set_node_inputs(div_node, &[sqrt_input]);
                    graph.storage.node_hdr[div_node as usize].op_tag = OpKind::Rsqrt as u8;
                } else {
                    // a / sqrt(b) → Mul(a, Rsqrt(b))：新建 Rsqrt 节点吃 b，Div 改 Mul 吃 [a, rsqrt_out]
                    let rsqrt_node = graph.add_node(OpKind::Rsqrt);
                    let rsqrt_out = graph.add_value(
                        type_of_value(graph, sqrt_input)?,
                        Some("rsqrt"),
                        rsqrt_node,
                    );
                    graph.storage.set_node_inputs(rsqrt_node, &[sqrt_input]);
                    graph.storage.set_node_outputs(rsqrt_node, &[rsqrt_out]);
                    graph
                        .storage
                        .set_node_inputs(div_node, &[numerator, rsqrt_out]);
                    graph.storage.node_hdr[div_node as usize].op_tag = OpKind::Mul as u8;
                }
                applied += 1;
            }
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
            FloatOpt::ReciprocalSqrt {
                recip_node,
                sqrt_node: _,
                sqrt_input,
            } => {
                // Reciprocal(Sqrt(x)) → Rsqrt(x)：把 Reciprocal 节点本身改成 Rsqrt，
                // 输入换成 x。输出 value 不变（使用者无感），Sqrt 变孤儿交给 DCE
                graph.storage.set_node_inputs(recip_node, &[sqrt_input]);
                graph.storage.node_hdr[recip_node as usize].op_tag = OpKind::Rsqrt as u8;
                applied += 1;
            }
            FloatOpt::DivByReciprocal {
                div_node,
                numerator,
                recip_input,
            } => {
                // a / Reciprocal(b) → Mul(a, b)：把 Div 节点改成 Mul，输入换成 [a, b]。
                // 输出 value 不变，Reciprocal 节点变孤儿交给 DCE
                graph
                    .storage
                    .set_node_inputs(div_node, &[numerator, recip_input]);
                graph.storage.node_hdr[div_node as usize].op_tag = OpKind::Mul as u8;
                applied += 1;
            }
            FloatOpt::SoftmaxOnline { .. } => {
                // 仅识别，不改图（online-softmax 是 kernel tiling 策略，非 IR 重写）
            }
        }
    }

    Ok(applied)
}

/// 取 value 的 Type（标量或张量），用于新建同型 value
fn type_of_value(graph: &Graph, v: ValueId) -> Result<base::Type> {
    let val = graph.value(v)?;
    let dtype = val.dtype();
    if val.is_tensor() {
        Ok(base::Type::Tensor {
            dtype,
            dims: val.shape().to_vec(),
        })
    } else {
        Ok(base::Type::Scalar(dtype))
    }
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

    /// `1.0 / sqrt(x)` → 单个 Rsqrt(x)：Div 节点本身改 Rsqrt，输入换成 x
    #[test]
    fn fast_inv_sqrt_one_becomes_rsqrt() {
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

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        // Div 节点应已变成 Rsqrt，输入换成 [x]
        let n = g.node(div).unwrap();
        assert_eq!(n.kind, OpKind::Rsqrt, "1.0/sqrt(x) 的 Div 应改成 Rsqrt");
        assert_eq!(n.inputs(), &[x], "Rsqrt 输入应为原 Sqrt 的输入 x");
        // 输出 value 仍是 out（使用者无感）
        assert_eq!(n.outputs(), &[out]);
    }

    /// `a / sqrt(b)`（a 非常量）→ `Mul(a, Rsqrt(b))`：新建 Rsqrt，Div 改 Mul
    #[test]
    fn fast_inv_sqrt_general_becomes_mul_rsqrt() {
        let mut g = Graph::new("test");
        let a = g.add_input(Type::Scalar(DType::F32), Some("a"));
        let b = g.add_input(Type::Scalar(DType::F32), Some("b"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let sqrt_out = g.add_value(Type::Scalar(DType::F32), Some("sx"), sqrt);
        g.storage.set_node_inputs(sqrt, &[b]);
        g.storage.set_node_outputs(sqrt, &[sqrt_out]);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        g.storage.set_node_inputs(div, &[a, sqrt_out]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        // Div 应改成 Mul，输入 [a, rsqrt_out]
        let n = g.node(div).unwrap();
        assert_eq!(n.kind, OpKind::Mul, "a/sqrt(b) 的 Div 应改成 Mul");
        assert_eq!(n.inputs()[0], a, "Mul 第一个输入应是原分子 a");
        // 第二个输入是新建的 Rsqrt 节点的输出
        let rsqrt_out = n.inputs()[1];
        let rsqrt_def = g.value(rsqrt_out).unwrap().def_node();
        let rsqrt_node = g.node(rsqrt_def).unwrap();
        assert_eq!(rsqrt_node.kind, OpKind::Rsqrt, "应新建 Rsqrt 节点");
        assert_eq!(rsqrt_node.inputs(), &[b], "Rsqrt 输入应为原 Sqrt 的输入 b");
    }

    /// `2.0 / sqrt(x)`（分子常量≠1）→ `Mul(2.0, Rsqrt(x))`，不折叠常量
    #[test]
    fn fast_inv_sqrt_const_numerator_not_one() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let sqrt_out = g.add_value(Type::Scalar(DType::F32), Some("sx"), sqrt);
        g.storage.set_node_inputs(sqrt, &[x]);
        g.storage.set_node_outputs(sqrt, &[sqrt_out]);
        let (_c, two) = g.add_constant_f64(2.0);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        g.storage.set_node_inputs(div, &[two, sqrt_out]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(div).unwrap();
        assert_eq!(n.kind, OpKind::Mul, "2.0/sqrt(x) 的 Div 应改成 Mul");
        // 第一个输入仍是常量 2.0
        assert_eq!(constant_value(&g, n.inputs()[0]).unwrap(), Some(2.0));
    }

    /// `sqrt(x) / a`（Sqrt 是分子不是除数）不应触发 FastInvSqrt
    #[test]
    fn sqrt_as_numerator_not_matched() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let sqrt_out = g.add_value(Type::Scalar(DType::F32), Some("sx"), sqrt);
        g.storage.set_node_inputs(sqrt, &[x]);
        g.storage.set_node_outputs(sqrt, &[sqrt_out]);
        let (_c, two) = g.add_constant_f64(2.0);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        // sqrt_out 是分子（ins[0]），2.0 是除数（ins[1]）→ 应触发 DivByConstToMul 而非 FastInvSqrt
        g.storage.set_node_inputs(div, &[sqrt_out, two]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            !opts
                .iter()
                .any(|o| matches!(o, FloatOpt::FastInvSqrt { .. })),
            "sqrt(x)/a 不应识别为 FastInvSqrt（除数不是 Sqrt）"
        );
    }

    /// RMSNorm 张量模式：`x / sqrt(y)`（张量）→ `Mul(x, Rsqrt(y))`，shape 正确传递
    #[test]
    fn fast_inv_sqrt_tensor_preserves_shape() {
        let mut g = Graph::new("test");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![2, 3],
        };
        let x = g.add_input(ty.clone(), Some("x"));
        let y = g.add_input(ty.clone(), Some("y"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let sqrt_out = g.add_value(ty.clone(), Some("sx"), sqrt);
        g.storage.set_node_inputs(sqrt, &[y]);
        g.storage.set_node_outputs(sqrt, &[sqrt_out]);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(ty.clone(), Some("out"), div);
        g.storage.set_node_inputs(div, &[x, sqrt_out]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(div).unwrap();
        assert_eq!(n.kind, OpKind::Mul);
        let rsqrt_out = n.inputs()[1];
        let rsqrt_def = g.value(rsqrt_out).unwrap().def_node();
        let rsqrt_node = g.node(rsqrt_def).unwrap();
        assert_eq!(rsqrt_node.kind, OpKind::Rsqrt);
        assert_eq!(rsqrt_node.inputs(), &[y]);
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

    /// `Reciprocal(Sqrt(x))` → `Rsqrt(x)`（2 op 降 1 op）：Reciprocal 节点本身改 Rsqrt
    #[test]
    fn reciprocal_sqrt_becomes_rsqrt() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let sqrt_out = g.add_value(Type::Scalar(DType::F32), Some("sx"), sqrt);
        g.storage.set_node_inputs(sqrt, &[x]);
        g.storage.set_node_outputs(sqrt, &[sqrt_out]);
        let recip = g.add_node(OpKind::Reciprocal);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), recip);
        g.storage.set_node_inputs(recip, &[sqrt_out]);
        g.storage.set_node_outputs(recip, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        // Reciprocal 节点应已变成 Rsqrt，输入换成 [x]
        let n = g.node(recip).unwrap();
        assert_eq!(n.kind, OpKind::Rsqrt, "Reciprocal(Sqrt(x)) 应改成 Rsqrt");
        assert_eq!(n.inputs(), &[x], "Rsqrt 输入应为原 Sqrt 的输入 x");
        assert_eq!(n.outputs(), &[out]);
    }

    /// `Reciprocal(x)`（输入非 Sqrt）不应触发 ReciprocalSqrt
    #[test]
    fn reciprocal_non_sqrt_not_matched() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let recip = g.add_node(OpKind::Reciprocal);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), recip);
        g.storage.set_node_inputs(recip, &[x]);
        g.storage.set_node_outputs(recip, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            !opts
                .iter()
                .any(|o| matches!(o, FloatOpt::ReciprocalSqrt { .. })),
            "Reciprocal(x) 输入非 Sqrt 不应触发 ReciprocalSqrt"
        );
    }

    /// `a / Reciprocal(b)` → `Mul(a, b)`（消去 Reciprocal+Div 换便宜 Mul）
    #[test]
    fn div_by_reciprocal_becomes_mul() {
        let mut g = Graph::new("test");
        let a = g.add_input(Type::Scalar(DType::F32), Some("a"));
        let b = g.add_input(Type::Scalar(DType::F32), Some("b"));
        let recip = g.add_node(OpKind::Reciprocal);
        let recip_out = g.add_value(Type::Scalar(DType::F32), Some("ro"), recip);
        g.storage.set_node_inputs(recip, &[b]);
        g.storage.set_node_outputs(recip, &[recip_out]);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        g.storage.set_node_inputs(div, &[a, recip_out]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(div).unwrap();
        assert_eq!(n.kind, OpKind::Mul, "a/Reciprocal(b) 的 Div 应改成 Mul");
        assert_eq!(n.inputs(), &[a, b], "Mul 输入应为 [a, b]");
    }

    /// `Reciprocal(a) / b`（Reciprocal 是分子不是除数）不应触发 DivByReciprocal
    #[test]
    fn reciprocal_as_numerator_not_matched() {
        let mut g = Graph::new("test");
        let a = g.add_input(Type::Scalar(DType::F32), Some("a"));
        let b = g.add_input(Type::Scalar(DType::F32), Some("b"));
        let recip = g.add_node(OpKind::Reciprocal);
        let recip_out = g.add_value(Type::Scalar(DType::F32), Some("ro"), recip);
        g.storage.set_node_inputs(recip, &[a]);
        g.storage.set_node_outputs(recip, &[recip_out]);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        // recip_out 是分子（ins[0]），b 是除数（ins[1]）→ 不应触发 DivByReciprocal
        g.storage.set_node_inputs(div, &[recip_out, b]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            !opts
                .iter()
                .any(|o| matches!(o, FloatOpt::DivByReciprocal { .. })),
            "Reciprocal(a)/b 不应触发 DivByReciprocal（除数不是 Reciprocal）"
        );
    }
}
