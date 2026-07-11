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
//! - **ExpMulFusion 重排**：`Exp(x) * Exp(y)` → `Exp(x + y)`（省一个 Exp）。
//!   恒等式 `e^x·e^y = e^(x+y)`。softmax/attention 里 exp 链相乘极常见，exp 是超越函数
//!   贵，重排成单个 Exp + 便宜 Add。浮点代数重排（类 online-softmax 式），非贪心模式。
//! - **ExpDivFusion 重排**：`Exp(x) / Exp(y)` → `Exp(x - y)`（省一个 Exp）。
//!   恒等式 `e^x/e^y = e^(x-y)`，幂除法法则，ExpMulFusion 的对偶。attention 里
//!   attention score 归一化（exp(score)/sum(exp)）常见此模式。
//! - **PowHalfToSqrt**：`Pow(x, 0.5)` → `Sqrt(x)` / `Pow(x, -0.5)` → `Rsqrt(x)`。
//!   把通用 Pow（log/exp 实现的超越函数，贵）换成专用单条硬件指令（IEEE754 sqrt/rsqrt，
//!   rsqrt 可用 0x5f3759df bit trick）。幂指数 ±0.5 时 x^0.5=√x，x^(-0.5)=1/√x。
//!   RMSNorm 的 `x * Pow(var+eps, -0.5)` 常见此模式
//! - **PowNegOneToReciprocal**：`Pow(x, -1.0)` → `Reciprocal(x)`。x^(-1)=1/x，
//!   同 PowHalfToSqrt 一类：通用幂 → 专用 op（单条硬件指令）
//! - **PowSquareToMul**：`Pow(x, 2.0)` → `Mul(x, x)`。x²=x·x，通用幂（用 log/exp
//!   实现的超越函数，贵）换成便宜乘法。无溢出/NaN 风险（数学完全等价）
//! - **SqrtSquareToAbs**：`Sqrt(x*x)` → `Abs(x)`。√(x²)=|x|，Sqrt+Mul 换单条
//!   Abs 硬件指令。两输入相同（x*x）才匹配，x*y 无简化。L2 norm 的 sqrt(x·x) 常见
//! - **LogExpToIdentity**：`Log(Exp(x))` → `x`。ln(eˣ)=x，消去 Log+Exp 两个 op。
//!   代数恒等式重写，ML 场景 x 极少大到 Exp 溢出（f32 阈值约 x>889），默认启用
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
    /// `Exp(x) * Exp(y)` → `Exp(x + y)`（省一个 Exp）。e^x·e^y = e^(x+y)
    ExpMulFusion {
        mul_node: base::NodeId,
        exp_x: base::NodeId,
        exp_y: base::NodeId,
        x_input: ValueId,
        y_input: ValueId,
    },
    /// `Exp(x) / Exp(y)` → `Exp(x - y)`（省一个 Exp）。e^x/e^y = e^(x-y)，ExpMulFusion 对偶
    ExpDivFusion {
        div_node: base::NodeId,
        exp_x: base::NodeId,
        exp_y: base::NodeId,
        x_input: ValueId,
        y_input: ValueId,
    },
    /// `Pow(x, 0.5)` → `Sqrt(x)` / `Pow(x, -0.5)` → `Rsqrt(x)`。
    /// 把通用 Pow（用 log/exp 实现的超越函数，贵）换成专用单条硬件指令
    /// （IEEE754 sqrt/rsqrt，rsqrt 可用 0x5f3759df bit trick）。幂指数为 ±0.5 时
    /// x^0.5=√x，x^(-0.5)=1/√x=rsqrt(x)。RMSNorm 的 `x * Pow(var+eps, -0.5)` 常见此模式
    PowHalfToSqrt {
        pow_node: base::NodeId,
        base: ValueId,
        /// true=指数 -0.5（→Rsqrt）；false=指数 0.5（→Sqrt）
        is_negative: bool,
    },
    /// `Pow(x, -1.0)` → `Reciprocal(x)`。x^(-1)=1/x，把通用 Pow 换成专用 Reciprocal
    /// （单条硬件指令 / 0x5f3759df 同族 bit trick）。同 PowHalfToSqrt 一类：通用幂 → 专用 op
    PowNegOneToReciprocal {
        pow_node: base::NodeId,
        base: ValueId,
    },
    /// `Sqrt(x*x)` → `Abs(x)`：√(x²)=|x|，Sqrt+Mul 换单条 Abs 硬件指令。
    /// 注意：只匹配两输入相同（x*x），不是 x*y（x≠y 时 √(xy) 无简化）
    SqrtSquareToAbs {
        sqrt_node: base::NodeId, // 主操作节点，改 Abs
        x_input: ValueId,        // Mul 的输入（两输入相同，取一个）
    },
    /// `Log(Exp(x))` → `x`：ln(eˣ)=x，消去 Log+Exp。log_node 仅记录（孤儿交 DCE）。
    /// 溢出边界：数学上 ln(eˣ)=x 对所有有限 x 成立。运行时 Exp(x) 溢出成 inf 时
    /// Log(inf)=inf，重写后直接得 x（有限值）——语义有差异。但这是代数恒等式重写，
    /// 且 ML 场景 x 极少大到 Exp 溢出（f32 阈值约 x>889）。默认启用。
    LogExpToIdentity {
        log_node: base::NodeId, // 仅记录，不改 op（输出被重写走，孤儿交 DCE）
        x_input: ValueId,       // Exp 的输入，重写目标
    },
    /// `Pow(x, 2.0)` → `Mul(x, x)`：x²=x·x，通用幂换便宜乘法（Pow 用 log/exp 实现，贵）
    PowSquareToMul {
        pow_node: base::NodeId, // 主操作节点，改 Mul
        base: ValueId,          // 底数 x，两个输入都用它
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
                if let Some(opt) = try_match_exp_div(graph, n)? {
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
                if let Some(opt) = try_match_exp_mul(graph, n)? {
                    opts.push(opt);
                }
            }
            OpKind::Reciprocal => {
                if let Some(opt) = try_match_reciprocal_sqrt(graph, n)? {
                    opts.push(opt);
                }
            }
            OpKind::Pow => {
                if let Some(opt) = try_match_pow_half(graph, n)? {
                    opts.push(opt);
                }
                if let Some(opt) = try_match_pow_square(graph, n)? {
                    opts.push(opt);
                }
            }
            OpKind::Sqrt => {
                if let Some(opt) = try_match_sqrt_square(graph, n)? {
                    opts.push(opt);
                }
            }
            OpKind::Log => {
                if let Some(opt) = try_match_log_exp(graph, n)? {
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

/// 识别 `Exp(x) * Exp(y)`：Mul 节点，两个输入都是 Exp 节点输出。
/// `e^x·e^y = e^(x+y)`，省一个 Exp（超越函数贵）。重排成 Exp(x+y) = Exp(Add(x,y))
/// 注意：两个 Exp 必须是不同节点（x==y 时是 exp(x)²，不能融成 exp(2x) 的本规则，
/// 那是另一类——但 exp(x)*exp(x)=exp(2x) 也成立，此处也覆盖：x==y 时 Add(x,x)=2x）
fn try_match_exp_mul(graph: &Graph, mul: NodeView) -> Result<Option<FloatOpt>> {
    let ins = mul.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (a, b) = (ins[0], ins[1]);
    let a_def = graph.value(a)?.def_node();
    let b_def = graph.value(b)?.def_node();
    if a_def == u32::MAX || b_def == u32::MAX {
        return Ok(None);
    }
    let a_node = graph.node(a_def)?;
    let b_node = graph.node(b_def)?;
    if a_node.kind != OpKind::Exp || b_node.kind != OpKind::Exp {
        return Ok(None);
    }
    let Some(&x_input) = a_node.inputs().first() else {
        return Ok(None);
    };
    let Some(&y_input) = b_node.inputs().first() else {
        return Ok(None);
    };
    Ok(Some(FloatOpt::ExpMulFusion {
        mul_node: mul.id,
        exp_x: a_def,
        exp_y: b_def,
        x_input,
        y_input,
    }))
}

/// 识别 `Exp(x) / Exp(y)`：Div 节点，两个输入都是 Exp 节点输出。
/// `e^x/e^y = e^(x-y)`，幂除法法则，ExpMulFusion 的对偶。省一个 Exp。
/// 注意：除数(ins[1])必须是 Exp，分子(ins[0])也必须是 Exp（顺序不可换，e^x/e^y≠e^y/e^x）
fn try_match_exp_div(graph: &Graph, div: NodeView) -> Result<Option<FloatOpt>> {
    let ins = div.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (a, b) = (ins[0], ins[1]);
    let a_def = graph.value(a)?.def_node();
    let b_def = graph.value(b)?.def_node();
    if a_def == u32::MAX || b_def == u32::MAX {
        return Ok(None);
    }
    let a_node = graph.node(a_def)?;
    let b_node = graph.node(b_def)?;
    if a_node.kind != OpKind::Exp || b_node.kind != OpKind::Exp {
        return Ok(None);
    }
    let Some(&x_input) = a_node.inputs().first() else {
        return Ok(None);
    };
    let Some(&y_input) = b_node.inputs().first() else {
        return Ok(None);
    };
    Ok(Some(FloatOpt::ExpDivFusion {
        div_node: div.id,
        exp_x: a_def,
        exp_y: b_def,
        x_input,
        y_input,
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

/// 识别 `Pow(x, 0.5)` / `Pow(x, -0.5)`：Pow 节点，指数输入(ins[1])是常量 ±0.5。
/// x^0.5=√x，x^(-0.5)=1/√x=rsqrt(x)。把通用 Pow 换成专用 sqrt/rsqrt 硬件指令。
/// 注意：只匹配指数是 ±0.5 常量（底数任意）。底数是常量时 algebra 的常量折叠已处理
fn try_match_pow_half(graph: &Graph, pow: NodeView) -> Result<Option<FloatOpt>> {
    let ins = pow.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (base, exp) = (ins[0], ins[1]);
    match constant_value(graph, exp)? {
        Some(0.5) => Ok(Some(FloatOpt::PowHalfToSqrt {
            pow_node: pow.id,
            base,
            is_negative: false,
        })),
        Some(-0.5) => Ok(Some(FloatOpt::PowHalfToSqrt {
            pow_node: pow.id,
            base,
            is_negative: true,
        })),
        Some(-1.0) => Ok(Some(FloatOpt::PowNegOneToReciprocal {
            pow_node: pow.id,
            base,
        })),
        _ => Ok(None),
    }
}

/// 识别 `Sqrt(x*x)`：Sqrt 节点，输入(ins[0]) 是 Mul 节点输出，且 Mul 两输入相同。
/// √(x²)=|x|，Sqrt+Mul 换单条 Abs 硬件指令。注意：只匹配 x*x（两输入相同），
/// x*y（x≠y）时 √(xy) 无简化。底数符号不影响（x² 始终非负，√(x²)=|x| 对所有 x 成立）
fn try_match_sqrt_square(graph: &Graph, sqrt: NodeView) -> Result<Option<FloatOpt>> {
    let Some(&y) = sqrt.inputs().first() else {
        return Ok(None);
    };
    let y_def = graph.value(y)?.def_node();
    if y_def == u32::MAX {
        return Ok(None);
    }
    let mul_node = graph.node(y_def)?;
    if mul_node.kind != OpKind::Mul {
        return Ok(None);
    }
    let ins = mul_node.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    // 两输入必须是同一个 value（x*x），不是 x*y
    if ins[0] != ins[1] {
        return Ok(None);
    }
    Ok(Some(FloatOpt::SqrtSquareToAbs {
        sqrt_node: sqrt.id,
        x_input: ins[0],
    }))
}

/// 识别 `Log(Exp(x))`：Log 节点，输入(ins[0]) 是 Exp 节点输出。
/// ln(eˣ)=x，消去 Log+Exp。Log 节点输出被重写到 x，节点本身不改 op（孤儿交 DCE）
fn try_match_log_exp(graph: &Graph, log: NodeView) -> Result<Option<FloatOpt>> {
    let Some(&y) = log.inputs().first() else {
        return Ok(None);
    };
    let y_def = graph.value(y)?.def_node();
    if y_def == u32::MAX {
        return Ok(None);
    }
    let exp_node = graph.node(y_def)?;
    if exp_node.kind != OpKind::Exp {
        return Ok(None);
    }
    let Some(&x_input) = exp_node.inputs().first() else {
        return Ok(None);
    };
    Ok(Some(FloatOpt::LogExpToIdentity {
        log_node: log.id,
        x_input,
    }))
}

/// 识别 `Pow(x, 2.0)`：Pow 节点，指数输入(ins[1])是常量 2.0。
/// x²=x·x，把通用 Pow（log/exp 实现的超越函数，贵）换成便宜 Mul。无溢出/NaN 风险
fn try_match_pow_square(graph: &Graph, pow: NodeView) -> Result<Option<FloatOpt>> {
    let ins = pow.inputs();
    if ins.len() != 2 {
        return Ok(None);
    }
    let (base, exp) = (ins[0], ins[1]);
    if constant_value(graph, exp)? == Some(2.0) {
        return Ok(Some(FloatOpt::PowSquareToMul {
            pow_node: pow.id,
            base,
        }));
    }
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
            FloatOpt::ExpMulFusion {
                mul_node,
                exp_x,
                exp_y: _,
                x_input,
                y_input,
            } => {
                // Exp(x)*Exp(y) → Exp(Add(x,y))：新建 Add 节点吃 [x,y]，复用 exp_x 节点
                // 改吃 Add 输出（op 不变仍是 Exp），Mul 节点变孤儿交给 DCE。
                // 结果用 exp_x 的输出 value（仍指向 exp_x），值等于原 Mul 输出，
                // 故把 Mul 输出的所有使用者重写到 exp_x 的输出
                let add_node = graph.add_node(OpKind::Add);
                let add_out =
                    graph.add_value(type_of_value(graph, x_input)?, Some("sum"), add_node);
                graph.storage.set_node_inputs(add_node, &[x_input, y_input]);
                graph.storage.set_node_outputs(add_node, &[add_out]);
                // exp_x 节点改吃 Add 输出
                graph.storage.set_node_inputs(exp_x, &[add_out]);
                // 把 Mul 输出的使用者重写到 exp_x 的输出（值相等）
                let mul_outs: Vec<ValueId> = graph.node(mul_node)?.outputs().to_vec();
                let exp_x_outs: Vec<ValueId> = graph.node(exp_x)?.outputs().to_vec();
                if let (Some(&exp_out), Some(&mul_out)) = (exp_x_outs.first(), mul_outs.first()) {
                    let node_ids: Vec<u32> = graph.node_ids().collect();
                    for nid in node_ids {
                        let old_inputs: Vec<ValueId> = graph.node(nid)?.inputs().to_vec();
                        if old_inputs.contains(&mul_out) {
                            let new_inputs: Vec<ValueId> = old_inputs
                                .iter()
                                .map(|&v| if v == mul_out { exp_out } else { v })
                                .collect();
                            graph.storage.set_node_inputs(nid, &new_inputs);
                        }
                    }
                    let old_outputs: Vec<ValueId> = graph.outputs().to_vec();
                    if old_outputs.contains(&mul_out) {
                        let new_outputs: Vec<ValueId> = old_outputs
                            .iter()
                            .map(|&v| if v == mul_out { exp_out } else { v })
                            .collect();
                        graph.storage.outputs = new_outputs;
                    }
                }
                applied += 1;
            }
            FloatOpt::ExpDivFusion {
                div_node,
                exp_x,
                exp_y: _,
                x_input,
                y_input,
            } => {
                // Exp(x)/Exp(y) → Exp(Sub(x,y))：新建 Sub 节点吃 [x,y]，复用 exp_x 节点
                // 改吃 Sub 输出，Div 输出使用者重写到 exp_x 输出（值相等）
                let sub_node = graph.add_node(OpKind::Sub);
                let sub_out =
                    graph.add_value(type_of_value(graph, x_input)?, Some("diff"), sub_node);
                graph.storage.set_node_inputs(sub_node, &[x_input, y_input]);
                graph.storage.set_node_outputs(sub_node, &[sub_out]);
                graph.storage.set_node_inputs(exp_x, &[sub_out]);
                let div_outs: Vec<ValueId> = graph.node(div_node)?.outputs().to_vec();
                let exp_x_outs: Vec<ValueId> = graph.node(exp_x)?.outputs().to_vec();
                if let (Some(&exp_out), Some(&div_out)) = (exp_x_outs.first(), div_outs.first()) {
                    let node_ids: Vec<u32> = graph.node_ids().collect();
                    for nid in node_ids {
                        let old_inputs: Vec<ValueId> = graph.node(nid)?.inputs().to_vec();
                        if old_inputs.contains(&div_out) {
                            let new_inputs: Vec<ValueId> = old_inputs
                                .iter()
                                .map(|&v| if v == div_out { exp_out } else { v })
                                .collect();
                            graph.storage.set_node_inputs(nid, &new_inputs);
                        }
                    }
                    let old_outputs: Vec<ValueId> = graph.outputs().to_vec();
                    if old_outputs.contains(&div_out) {
                        let new_outputs: Vec<ValueId> = old_outputs
                            .iter()
                            .map(|&v| if v == div_out { exp_out } else { v })
                            .collect();
                        graph.storage.outputs = new_outputs;
                    }
                }
                applied += 1;
            }
            FloatOpt::SoftmaxOnline { .. } => {
                // 仅识别，不改图（online-softmax 是 kernel tiling 策略，非 IR 重写）
            }
            FloatOpt::PowHalfToSqrt {
                pow_node,
                base,
                is_negative,
            } => {
                // Pow(x, 0.5) → Sqrt(x) / Pow(x, -0.5) → Rsqrt(x)：
                // 把 Pow 节点 op 改成 Sqrt/Rsqrt，输入换成 [x]（丢弃常量指数）。
                // 输出 value 不变（使用者无感），常量指数变孤儿交给 DCE
                graph.storage.set_node_inputs(pow_node, &[base]);
                graph.storage.node_hdr[pow_node as usize].op_tag = if is_negative {
                    OpKind::Rsqrt as u8
                } else {
                    OpKind::Sqrt as u8
                };
                applied += 1;
            }
            FloatOpt::PowNegOneToReciprocal { pow_node, base } => {
                // Pow(x, -1.0) → Reciprocal(x)：把 Pow 节点 op 改成 Reciprocal，
                // 输入换成 [x]（丢弃常量指数）。x^(-1)=1/x=reciprocal(x)
                graph.storage.set_node_inputs(pow_node, &[base]);
                graph.storage.node_hdr[pow_node as usize].op_tag = OpKind::Reciprocal as u8;
                applied += 1;
            }
            FloatOpt::SqrtSquareToAbs { sqrt_node, x_input } => {
                // Sqrt(x*x) → Abs(x)：√(x²)=|x|。把 Sqrt 节点 op 改成 Abs，
                // 输入换成 [x]（丢弃 Mul 的另一份相同输入）。输出 value 不变（使用者无感），
                // 原 Mul(x,x) 节点变孤儿交给 DCE
                graph.storage.set_node_inputs(sqrt_node, &[x_input]);
                graph.storage.node_hdr[sqrt_node as usize].op_tag = OpKind::Abs as u8;
                applied += 1;
            }
            FloatOpt::LogExpToIdentity { log_node, x_input } => {
                // Log(Exp(x)) → x：ln(eˣ)=x。把 Log 输出值的所有使用者重写为 x_input
                // （Exp 的输入）。Log 节点本身不改 op，输出无人用后变孤儿交给 DCE
                let log_outs: Vec<ValueId> = graph.node(log_node)?.outputs().to_vec();
                if let Some(&log_out) = log_outs.first() {
                    if log_out != x_input {
                        // 避免自引用死循环：log_out != x_input 才重写
                        let node_ids: Vec<u32> = graph.node_ids().collect();
                        for nid in node_ids {
                            let old_inputs: Vec<ValueId> = graph.node(nid)?.inputs().to_vec();
                            if old_inputs.contains(&log_out) {
                                let new_inputs: Vec<ValueId> = old_inputs
                                    .iter()
                                    .map(|&v| if v == log_out { x_input } else { v })
                                    .collect();
                                graph.storage.set_node_inputs(nid, &new_inputs);
                            }
                        }
                        // 图输出也要重写
                        let old_outputs: Vec<ValueId> = graph.outputs().to_vec();
                        if old_outputs.contains(&log_out) {
                            let new_outputs: Vec<ValueId> = old_outputs
                                .iter()
                                .map(|&v| if v == log_out { x_input } else { v })
                                .collect();
                            graph.storage.outputs = new_outputs;
                        }
                    }
                }
                applied += 1;
            }
            FloatOpt::PowSquareToMul { pow_node, base } => {
                // Pow(x, 2.0) → Mul(x, x)：Pow 节点改 Mul，输入换 [x, x]（两输入都是 base），
                // 丢弃常量指数 2.0。输出 value 不变（使用者无感），原常量 2.0 节点变孤儿交 DCE
                graph.storage.set_node_inputs(pow_node, &[base, base]);
                graph.storage.node_hdr[pow_node as usize].op_tag = OpKind::Mul as u8;
                applied += 1;
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

    /// `Exp(x) * Exp(y)` → `Exp(Add(x,y))`（省一个 Exp）。e^x·e^y = e^(x+y)
    #[test]
    fn exp_mul_fuses_to_exp_add() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let y = g.add_input(Type::Scalar(DType::F32), Some("y"));
        let exp_x = g.add_node(OpKind::Exp);
        let ex_out = g.add_value(Type::Scalar(DType::F32), Some("ex"), exp_x);
        g.storage.set_node_inputs(exp_x, &[x]);
        g.storage.set_node_outputs(exp_x, &[ex_out]);
        let exp_y = g.add_node(OpKind::Exp);
        let ey_out = g.add_value(Type::Scalar(DType::F32), Some("ey"), exp_y);
        g.storage.set_node_inputs(exp_y, &[y]);
        g.storage.set_node_outputs(exp_y, &[ey_out]);
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), mul);
        g.storage.set_node_inputs(mul, &[ex_out, ey_out]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        // 应新建 1 个 Add 节点吃 [x,y]；复用的 exp_x 改吃 Add 输出。
        // exp_y 变孤儿（apply 不删，交给 DCE），此处不检查节点数，只验证重排正确。
        let add_count = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::Add)
            .count();
        assert_eq!(add_count, 1, "应新建 1 个 Add 节点");
        let add_node = g
            .node_ids()
            .find(|&id| g.node(id).unwrap().kind == OpKind::Add)
            .unwrap();
        assert_eq!(
            g.node(add_node).unwrap().inputs(),
            &[x, y],
            "Add 输入应是 [x, y]"
        );
        // exp_x 节点（复用为结果）输入应已换成 Add 输出
        let add_out = g.node(add_node).unwrap().outputs()[0];
        assert_eq!(
            g.node(exp_x).unwrap().inputs(),
            &[add_out],
            "复用的 exp_x 输入应是 Add 输出"
        );
        // Mul 的输出使用者应被重写到 exp_x 的输出（图输出 out 应指向 exp_x_out）
        assert!(g.outputs().contains(&ex_out), "图输出应重写到 exp_x 的输出");
    }

    /// `Exp(x) / Exp(y)` → `Exp(Sub(x,y))`（省一个 Exp）。e^x/e^y = e^(x-y)
    #[test]
    fn exp_div_fuses_to_exp_sub() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let y = g.add_input(Type::Scalar(DType::F32), Some("y"));
        let exp_x = g.add_node(OpKind::Exp);
        let ex_out = g.add_value(Type::Scalar(DType::F32), Some("ex"), exp_x);
        g.storage.set_node_inputs(exp_x, &[x]);
        g.storage.set_node_outputs(exp_x, &[ex_out]);
        let exp_y = g.add_node(OpKind::Exp);
        let ey_out = g.add_value(Type::Scalar(DType::F32), Some("ey"), exp_y);
        g.storage.set_node_inputs(exp_y, &[y]);
        g.storage.set_node_outputs(exp_y, &[ey_out]);
        let div = g.add_node(OpKind::Div);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), div);
        g.storage.set_node_inputs(div, &[ex_out, ey_out]);
        g.storage.set_node_outputs(div, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        // 应新建 1 个 Sub 节点吃 [x,y]；复用的 exp_x 改吃 Sub 输出
        let sub_count = g
            .node_ids()
            .filter(|&id| g.node(id).unwrap().kind == OpKind::Sub)
            .count();
        assert_eq!(sub_count, 1, "应新建 1 个 Sub 节点");
        let sub_node = g
            .node_ids()
            .find(|&id| g.node(id).unwrap().kind == OpKind::Sub)
            .unwrap();
        assert_eq!(
            g.node(sub_node).unwrap().inputs(),
            &[x, y],
            "Sub 输入应是 [x, y]"
        );
        let sub_out = g.node(sub_node).unwrap().outputs()[0];
        assert_eq!(
            g.node(exp_x).unwrap().inputs(),
            &[sub_out],
            "复用的 exp_x 输入应是 Sub 输出"
        );
        assert!(g.outputs().contains(&ex_out), "图输出应重写到 exp_x 的输出");
    }

    /// `Pow(x, 0.5)` → `Sqrt(x)`：Pow 节点本身改 Sqrt，输入换成 [x]
    #[test]
    fn pow_half_becomes_sqrt() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, half) = g.add_constant_f64(0.5);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, half]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(pow).unwrap();
        assert_eq!(n.kind, OpKind::Sqrt, "Pow(x,0.5) 应改成 Sqrt");
        assert_eq!(n.inputs(), &[x], "Sqrt 输入应为 [x]");
        assert_eq!(n.outputs(), &[out]);
    }

    /// `Pow(x, -0.5)` → `Rsqrt(x)`：Pow 节点本身改 Rsqrt，输入换成 [x]
    #[test]
    fn pow_neg_half_becomes_rsqrt() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, neg_half) = g.add_constant_f64(-0.5);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, neg_half]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(pow).unwrap();
        assert_eq!(n.kind, OpKind::Rsqrt, "Pow(x,-0.5) 应改成 Rsqrt");
        assert_eq!(n.inputs(), &[x], "Rsqrt 输入应为 [x]");
        assert_eq!(n.outputs(), &[out]);
    }

    /// `Pow(x, 2.0)`（指数非 ±0.5）不应触发 PowHalfToSqrt
    #[test]
    fn pow_non_half_not_matched() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, two) = g.add_constant_f64(2.0);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, two]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            !opts
                .iter()
                .any(|o| matches!(o, FloatOpt::PowHalfToSqrt { .. })),
            "Pow(x,2.0) 不应触发 PowHalfToSqrt"
        );
    }

    /// `Pow(x, 0.5)` 张量模式：shape 正确传递（输出 value 不变）
    #[test]
    fn pow_half_tensor_preserves_shape() {
        let mut g = Graph::new("test");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![2, 3],
        };
        let x = g.add_input(ty.clone(), Some("x"));
        let (_c, half) = g.add_constant_f64(0.5);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(ty.clone(), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, half]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(pow).unwrap();
        assert_eq!(n.kind, OpKind::Sqrt);
        assert_eq!(n.inputs(), &[x]);
    }

    /// `Pow(x, -1.0)` → `Reciprocal(x)`：Pow 节点本身改 Reciprocal，输入换成 [x]
    #[test]
    fn pow_neg_one_becomes_reciprocal() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, neg_one) = g.add_constant_f64(-1.0);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, neg_one]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(pow).unwrap();
        assert_eq!(n.kind, OpKind::Reciprocal, "Pow(x,-1) 应改成 Reciprocal");
        assert_eq!(n.inputs(), &[x], "Reciprocal 输入应为 [x]");
        assert_eq!(n.outputs(), &[out]);
    }

    /// `Sqrt(x*x)` → `Abs(x)`：√(x²)=|x|，Sqrt 节点本身改 Abs，输入换成 [x]
    #[test]
    fn sqrt_square_to_abs() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let mul = g.add_node(OpKind::Mul);
        let mul_out = g.add_value(Type::Scalar(DType::F32), Some("m"), mul);
        g.storage.set_node_inputs(mul, &[x, x]); // x*x
        g.storage.set_node_outputs(mul, &[mul_out]);
        let sqrt = g.add_node(OpKind::Sqrt);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sqrt);
        g.storage.set_node_inputs(sqrt, &[mul_out]);
        g.storage.set_node_outputs(sqrt, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        // Sqrt 节点应已变成 Abs，输入换成 [x]
        let n = g.node(sqrt).unwrap();
        assert_eq!(n.kind, OpKind::Abs, "Sqrt(x*x) 应改成 Abs");
        assert_eq!(n.inputs(), &[x], "Abs 输入应为 [x]");
        assert_eq!(n.outputs(), &[out]);
    }

    /// `Sqrt(x*y)`（x≠y）不应触发 SqrtSquareToAbs（两输入不同，√(xy) 无简化）
    #[test]
    fn sqrt_xy_not_matched() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let y = g.add_input(Type::Scalar(DType::F32), Some("y"));
        let mul = g.add_node(OpKind::Mul);
        let mul_out = g.add_value(Type::Scalar(DType::F32), Some("m"), mul);
        g.storage.set_node_inputs(mul, &[x, y]); // x*y（x≠y）
        g.storage.set_node_outputs(mul, &[mul_out]);
        let sqrt = g.add_node(OpKind::Sqrt);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sqrt);
        g.storage.set_node_inputs(sqrt, &[mul_out]);
        g.storage.set_node_outputs(sqrt, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            !opts
                .iter()
                .any(|o| matches!(o, FloatOpt::SqrtSquareToAbs { .. })),
            "Sqrt(x*y)（x≠y）不应触发 SqrtSquareToAbs"
        );
    }

    /// `Sqrt(x*x)` 张量模式：改 Abs 后输出 value shape 不变
    #[test]
    fn sqrt_square_tensor_preserves_shape() {
        let mut g = Graph::new("test");
        let ty = Type::Tensor {
            dtype: DType::F32,
            dims: vec![2, 3],
        };
        let x = g.add_input(ty.clone(), Some("x"));
        let mul = g.add_node(OpKind::Mul);
        let mul_out = g.add_value(ty.clone(), Some("m"), mul);
        g.storage.set_node_inputs(mul, &[x, x]);
        g.storage.set_node_outputs(mul, &[mul_out]);
        let sqrt = g.add_node(OpKind::Sqrt);
        let out = g.add_value(ty.clone(), Some("out"), sqrt);
        g.storage.set_node_inputs(sqrt, &[mul_out]);
        g.storage.set_node_outputs(sqrt, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(sqrt).unwrap();
        assert_eq!(n.kind, OpKind::Abs);
        assert_eq!(n.inputs(), &[x]);
        // 输出 value 仍是 out，shape 不变
        assert_eq!(n.outputs(), &[out]);
        let out_val = g.value(out).unwrap();
        assert_eq!(out_val.shape(), &[2, 3], "输出 shape 应保持 [2,3]");
    }

    /// `Log(Exp(x))` → `x`：ln(eˣ)=x，图输出重写到 x_input，Exp 输入仍是 x
    #[test]
    fn log_exp_to_identity() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let exp = g.add_node(OpKind::Exp);
        let exp_out = g.add_value(Type::Scalar(DType::F32), Some("ex"), exp);
        g.storage.set_node_inputs(exp, &[x]);
        g.storage.set_node_outputs(exp, &[exp_out]);
        let log = g.add_node(OpKind::Log);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), log);
        g.storage.set_node_inputs(log, &[exp_out]);
        g.storage.set_node_outputs(log, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        // 图输出应重写到 x（不再是 log 的输出 out）
        assert!(
            g.outputs().contains(&x),
            "图输出应重写到 x_input（Exp 的输入）"
        );
        assert!(!g.outputs().contains(&out), "log 的输出 out 不应再是图输出");
        // Exp 节点输入仍是 x（apply 不改 Exp）
        assert_eq!(g.node(exp).unwrap().inputs(), &[x], "Exp 节点输入应仍是 x");
    }

    /// `Log(Sqrt(x))`（输入非 Exp）不应触发 LogExpToIdentity
    #[test]
    fn log_non_exp_not_matched() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let sqrt_out = g.add_value(Type::Scalar(DType::F32), Some("sx"), sqrt);
        g.storage.set_node_inputs(sqrt, &[x]);
        g.storage.set_node_outputs(sqrt, &[sqrt_out]);
        let log = g.add_node(OpKind::Log);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), log);
        g.storage.set_node_inputs(log, &[sqrt_out]);
        g.storage.set_node_outputs(log, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            !opts
                .iter()
                .any(|o| matches!(o, FloatOpt::LogExpToIdentity { .. })),
            "Log(Sqrt(x)) 不应触发 LogExpToIdentity（输入非 Exp）"
        );
    }

    /// `Pow(x, 2.0)` → `Mul(x, x)`：Pow 节点本身改 Mul，输入换成 [x, x]
    #[test]
    fn pow_square_to_mul() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, two) = g.add_constant_f64(2.0);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, two]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);

        let count = apply_float_opts(&mut g).unwrap();
        assert_eq!(count, 1);
        let n = g.node(pow).unwrap();
        assert_eq!(n.kind, OpKind::Mul, "Pow(x,2.0) 应改成 Mul");
        assert_eq!(n.inputs(), &[x, x], "Mul 输入应为 [x, x]");
        assert_eq!(n.outputs(), &[out]);
    }

    /// `Pow(x, 3.0)`（指数非 2.0）不应触发 PowSquareToMul
    #[test]
    fn pow_non_two_not_matched() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, three) = g.add_constant_f64(3.0);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, three]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);

        let opts = find_opportunities(&g).unwrap();
        assert!(
            !opts
                .iter()
                .any(|o| matches!(o, FloatOpt::PowSquareToMul { .. })),
            "Pow(x,3.0) 不应触发 PowSquareToMul"
        );
    }
}
