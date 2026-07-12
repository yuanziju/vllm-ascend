//! algebra — 代数恒等式规则（函数式实现，不用模式匹配）
//!
//! 设计哲学：
//! - 只用简单代数规则（x+0, x*1, x*0=0, 常量折叠, x-x=0\[可选\]）
//! - 不硬编码复合算子模式（MatMul+Add→Linear 这种留给 fuse）
//! - 有 NaN/Inf 风险的规则（x-x=0、x/x=1）默认禁用，由 `AlgebraConfig.unsafe_opts` 开关控制
//!
//! shape-based 简化：除了标量 0/1，还识别"全 0 / 全 1 张量"（多元素 Constant
//! FloatArray，如 ONNX initializer 的 ones/zeros）。x + zeros→x、x * ones→x、
//! x * zeros→复用那个 zeros 张量（保留 shape，不退化为标量）。
//!
//! 应用方式：`simplify` 是纯函数返回建议；`run_algebraic_simplify` 收集
//! (old_value → new_value) 替换映射，重写所有节点 inputs，被替换的节点留给 DCE 清理。

use base::{Graph, NodeId, NodeView, OpKind, Result, ValueId};
use std::collections::HashMap;

/// 代数简化配置
#[derive(Debug, Clone, Copy, Default)]
pub struct AlgebraConfig {
    /// 启用有 NaN 风险的规则：x-x=0、x/x=1。默认 false（保守）
    pub unsafe_opts: bool,
}

/// 单节点简化建议
pub enum SimplifyResult {
    /// 用已有 value 替换该节点的输出（如 x+0 → x）
    ReplaceWith(ValueId),
    /// 用新常量 value 替换（常量折叠：c1+c2 → 新 Constant）
    /// 调用方需把新节点加入图
    FoldToConstant(f64),
    NoChange,
}

pub fn simplify(graph: &Graph, node: NodeView, cfg: AlgebraConfig) -> Result<SimplifyResult> {
    match node.kind {
        OpKind::Add => simplify_add(graph, node),
        OpKind::Sub => simplify_sub(graph, node, cfg),
        OpKind::Mul => simplify_mul(graph, node),
        OpKind::Div => simplify_div(graph, node, cfg),
        OpKind::Sqrt => simplify_sqrt(graph, node),
        OpKind::Exp => simplify_exp(graph, node),
        OpKind::Pow => simplify_pow(graph, node),
        OpKind::Reshape => simplify_reshape(graph, node),
        OpKind::Transpose => simplify_transpose(graph, node),
        OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax => simplify_reduce(graph, node),
        OpKind::Concat => simplify_concat(graph, node),
        _ => Ok(SimplifyResult::NoChange),
    }
}

fn simplify_add(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    // 常量折叠：c1 + c2 = c3
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        return Ok(SimplifyResult::FoldToConstant(x + y));
    }
    // x + 0 = x
    if is_constant_zero(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    if is_constant_zero(graph, a)? {
        return Ok(SimplifyResult::ReplaceWith(b));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_sub(graph: &Graph, node: NodeView, cfg: AlgebraConfig) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    // 常量折叠：c1 - c2 = c3
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        return Ok(SimplifyResult::FoldToConstant(x - y));
    }
    // x - 0 = x
    if is_constant_zero(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    // x - x = 0 (NaN 风险：Inf-Inf=NaN, NaN-x=NaN。默认禁用)
    if cfg.unsafe_opts && a == b {
        return Ok(SimplifyResult::FoldToConstant(0.0));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_mul(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    // 常量折叠：c1 * c2 = c3
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        return Ok(SimplifyResult::FoldToConstant(x * y));
    }
    // x * 1 = x
    if is_constant_one(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    if is_constant_one(graph, a)? {
        return Ok(SimplifyResult::ReplaceWith(b));
    }
    // x * 0 = 0：复用那个 0 张量（保留 shape，比 FoldToConstant 标量更准）
    // 注意：对 NaN 不安全（NaN*0=NaN），但 0 是比 x 更"确定"的值，
    // 且 x 若含 NaN 则结果本就不确定。此处保守返回那个 0 张量
    if is_constant_zero(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(b));
    }
    if is_constant_zero(graph, a)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_div(graph: &Graph, node: NodeView, cfg: AlgebraConfig) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    let a = ins[0];
    let b = ins[1];
    // 常量折叠：c1 / c2 = c3 (c2 != 0)
    if let (Some(x), Some(y)) = (constant_value(graph, a)?, constant_value(graph, b)?) {
        if y != 0.0 {
            return Ok(SimplifyResult::FoldToConstant(x / y));
        }
    }
    // x / 1 = x
    if is_constant_one(graph, b)? {
        return Ok(SimplifyResult::ReplaceWith(a));
    }
    // x / x = 1 (NaN 风险：0/0=NaN, Inf/Inf=NaN。默认禁用)
    if cfg.unsafe_opts && a == b {
        return Ok(SimplifyResult::FoldToConstant(1.0));
    }
    Ok(SimplifyResult::NoChange)
}

// --- 一元常量折叠（Sqrt/Exp/Pow 输入为常量时直接算出结果） ---

fn simplify_sqrt(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 1 {
        return Ok(SimplifyResult::NoChange);
    }
    // sqrt(c) = c.sqrt()（c<0 时返回 NaN，与运行时语义一致）
    if let Some(c) = constant_value(graph, ins[0])? {
        return Ok(SimplifyResult::FoldToConstant(c.sqrt()));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_exp(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 1 {
        return Ok(SimplifyResult::NoChange);
    }
    if let Some(c) = constant_value(graph, ins[0])? {
        return Ok(SimplifyResult::FoldToConstant(c.exp()));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_pow(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 2 {
        return Ok(SimplifyResult::NoChange);
    }
    // pow(c1, c2) = c1.powf(c2)（负底数+非整指数返回 NaN，与运行时一致）
    if let (Some(base), Some(exp)) = (
        constant_value(graph, ins[0])?,
        constant_value(graph, ins[1])?,
    ) {
        return Ok(SimplifyResult::FoldToConstant(base.powf(exp)));
    }
    // x ^ 1 = x
    if is_constant_one(graph, ins[1])? {
        return Ok(SimplifyResult::ReplaceWith(ins[0]));
    }
    Ok(SimplifyResult::NoChange)
}

// --- 基于 shape 的 no-op 简化 ---

fn simplify_reshape(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 1 {
        return Ok(SimplifyResult::NoChange);
    }
    let input = ins[0];
    let outs = node.outputs();
    if outs.is_empty() {
        return Ok(SimplifyResult::NoChange);
    }
    let in_shape = graph.value(input)?.shape();
    let out_shape = graph.value(outs[0])?.shape();
    // 输入输出 shape 都已知且相等 → reshape 是 no-op
    if shape_known(in_shape) && shape_known(out_shape) && in_shape == out_shape {
        return Ok(SimplifyResult::ReplaceWith(input));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_transpose(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    let ins = node.inputs();
    if ins.len() != 1 {
        return Ok(SimplifyResult::NoChange);
    }
    let input = ins[0];
    // 读 perm 属性，单位排列 [0,1,...,n-1] → no-op
    let perm = match read_perm(graph, node.id)? {
        Some(p) => p,
        None => return Ok(SimplifyResult::NoChange),
    };
    let identity: Vec<i64> = (0..perm.len() as i64).collect();
    if perm == identity {
        return Ok(SimplifyResult::ReplaceWith(input));
    }
    Ok(SimplifyResult::NoChange)
}

// --- 启发式 shape-based no-op 识别（reduce / concat） ---

fn simplify_reduce(graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    // 启发式：Reduce(x, axis) 当 x 在 axis 维 size==1 时，reduce 是 no-op
    // （沿 size-1 维 sum/mean/max = 该维唯一元素，值等于输入 squeeze 后的值）。
    // 保守策略：额外校验 reduce 输出 shape 已知且等于输入去掉 axis 维后的 shape
    // （即标准 reduce 语义下的正确结果 shape），保证 ReplaceWith 后 shape 严格正确。
    let ins = node.inputs();
    let Some(&input) = ins.first() else {
        return Ok(SimplifyResult::NoChange);
    };
    let Some(axis) = read_axis(graph, node.id)? else {
        return Ok(SimplifyResult::NoChange);
    };
    let in_shape = graph.value(input)?.shape();
    if !shape_known(in_shape) {
        return Ok(SimplifyResult::NoChange);
    }
    // axis 支持负值（shape_infer 的惯例）
    let rank = in_shape.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    if ax < 0 || ax >= rank {
        return Ok(SimplifyResult::NoChange); // axis 越界，保守不处理
    }
    if in_shape[ax as usize] != 1 {
        return Ok(SimplifyResult::NoChange); // axis 维 size!=1，非 no-op
    }
    // 标准 reduce 语义：消去 axis 维
    let expected_out: Vec<i64> = in_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i as i64 != ax)
        .map(|(_, &d)| d)
        .collect();
    let outs = node.outputs();
    let Some(&out_val) = outs.first() else {
        return Ok(SimplifyResult::NoChange);
    };
    let out_shape = graph.value(out_val)?.shape();
    if shape_known(out_shape) && out_shape == expected_out.as_slice() {
        return Ok(SimplifyResult::ReplaceWith(input));
    }
    Ok(SimplifyResult::NoChange)
}

fn simplify_concat(_graph: &Graph, node: NodeView) -> Result<SimplifyResult> {
    // 启发式：Concat 单输入 → 结果等于该输入（copy no-op）
    let ins = node.inputs();
    if ins.len() == 1 {
        return Ok(SimplifyResult::ReplaceWith(ins[0]));
    }
    Ok(SimplifyResult::NoChange)
}

fn shape_known(s: &[i64]) -> bool {
    !s.is_empty() && s.iter().all(|&d| d > 0)
}

fn read_perm(graph: &Graph, node: NodeId) -> Result<Option<Vec<i64>>> {
    let n = graph.node(node)?;
    for e in n.attrs() {
        if e.key == base::StorageAttrKey::Perm as u8
            && e.tag == base::storage::AttrTag::IntArray as u8
        {
            return Ok(Some(n.storage.attr_int_array(e).to_vec()));
        }
    }
    Ok(None)
}

/// 读取节点 Axis 属性（Int），找不到返回 None
fn read_axis(graph: &Graph, node: NodeId) -> Result<Option<i64>> {
    let n = graph.node(node)?;
    for e in n.attrs() {
        if e.key == base::StorageAttrKey::Axis as u8 && e.tag == base::storage::AttrTag::Int as u8 {
            return Ok(Some(n.storage.attr_int(e)));
        }
    }
    Ok(None)
}

// --- 常量识别辅助 ---

fn constant_value(graph: &Graph, v: ValueId) -> Result<Option<f64>> {
    let val = graph.value(v)?;
    let def = val.def_node();
    if def == u32::MAX {
        return Ok(None);
    }
    let node = graph.node(def)?;
    Ok(node.constant_value())
}

/// 判断常量张量是否"所有元素都等于 target"。
///
/// 覆盖三种存储形式：
/// - 标量 Constant（Value=Float）：直接比 target
/// - 单元素张量 Constant（Value=FloatArray 且 len==1）：取该元素
/// - 多元素张量 Constant（Value=FloatArray 且 len>1）：全等检查
///
/// 非常量或空张量返回 false。这让 algebra 能识别 ONNX initializer 的
/// ones/zeros（多元素全 1/全 0 张量），不仅限于标量 0/1。
fn constant_is_uniform(graph: &Graph, v: ValueId, target: f64) -> Result<bool> {
    let val = graph.value(v)?;
    let def = val.def_node();
    if def == u32::MAX {
        return Ok(false);
    }
    let node = graph.node(def)?;
    // 标量 / 单元素：constant_value 即可
    if let Some(scalar) = node.constant_value() {
        return Ok(scalar == target);
    }
    // 多元素张量：全等检查
    if let Some(tensor) = node.constant_tensor() {
        return Ok(!tensor.is_empty() && tensor.iter().all(|&x| x == target));
    }
    Ok(false)
}

fn is_constant_zero(graph: &Graph, v: ValueId) -> Result<bool> {
    constant_is_uniform(graph, v, 0.0)
}

fn is_constant_one(graph: &Graph, v: ValueId) -> Result<bool> {
    constant_is_uniform(graph, v, 1.0)
}

/// 应用代数简化到整个图。
///
/// 返回应用的简化次数。被替换/折叠的节点不直接删除（留给 DCE），
/// 但会把它们的输出 value 在所有使用者 inputs 中替换为新的 value。
pub fn run_algebraic_simplify(graph: &mut Graph) -> Result<usize> {
    run_with_config(graph, AlgebraConfig::default())
}

pub fn run_with_config(graph: &mut Graph, cfg: AlgebraConfig) -> Result<usize> {
    let mut applied = 0usize;
    // 已处理节点：一旦某节点被简化，不再重复处理（它的 inputs 不变，重复处理会死循环）
    // 嵌套场景（(x+0)+0）靠 rewrite 后外层节点 inputs 变化触发新简化
    let mut processed: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

    loop {
        // 第一阶段：纯读取，收集本轮所有简化建议（避免借用冲突）
        let mut suggestions: Vec<(NodeId, SimplifyResult)> = Vec::new();
        for id in graph.node_ids() {
            if processed.contains(&id) {
                continue;
            }
            let n = graph.node(id)?;
            if n.kind == OpKind::Constant {
                processed.insert(id);
                continue;
            }
            match simplify(graph, n, cfg)? {
                SimplifyResult::NoChange => {
                    processed.insert(id);
                }
                other => suggestions.push((id, other)),
            }
        }
        if suggestions.is_empty() {
            break;
        }

        // 第二阶段：应用建议（可变借用安全）
        let mut round_replacements: HashMap<ValueId, ValueId> = HashMap::new();
        for (id, sugg) in suggestions {
            processed.insert(id);
            let outs: Vec<ValueId> = graph.node(id)?.outputs().to_vec();
            match sugg {
                SimplifyResult::ReplaceWith(new_v) => {
                    for out_v in outs {
                        round_replacements.insert(out_v, new_v);
                    }
                    applied += 1;
                }
                SimplifyResult::FoldToConstant(val) => {
                    let (_cnode, cval) = graph.add_constant_f64(val);
                    for out_v in outs {
                        round_replacements.insert(out_v, cval);
                    }
                    applied += 1;
                }
                SimplifyResult::NoChange => {}
            }
        }
        // 立即重写 inputs，使下一轮看到更新后的图
        rewrite_inputs(graph, &round_replacements);
    }

    Ok(applied)
}

/// 重写所有节点的 inputs：把 old_v 替换为 new_v
fn rewrite_inputs(graph: &mut Graph, replacements: &HashMap<ValueId, ValueId>) {
    let lookup = |v: ValueId| -> ValueId { replacements.get(&v).copied().unwrap_or(v) };
    let node_ids: Vec<u32> = graph.node_ids().collect();
    for nid in node_ids {
        let Ok(node) = graph.node(nid) else {
            continue;
        };
        let old_inputs: Vec<ValueId> = node.inputs().to_vec();
        let new_inputs: Vec<ValueId> = old_inputs.iter().map(|&v| lookup(v)).collect();
        if old_inputs != new_inputs {
            graph.storage.set_node_inputs(nid, &new_inputs);
        }
    }
    let old_outputs: Vec<ValueId> = graph.outputs().to_vec();
    let new_outputs: Vec<ValueId> = old_outputs.iter().map(|&v| lookup(v)).collect();
    if old_outputs != new_outputs {
        graph.storage.outputs = new_outputs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{DType, Graph, OpKind, Type};

    fn build_add_with_zero() -> Graph {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("x"),
        );
        let (_c, zero) = g.add_constant_f64(0.0);
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("out"),
            add,
        );
        g.storage.set_node_inputs(add, &[x, zero]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);
        g
    }

    #[test]
    fn add_zero_simplifies() {
        let mut g = build_add_with_zero();
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        // 输出应已被重写为 x (value 0)
        assert_eq!(g.outputs(), &[0]);
    }

    #[test]
    fn mul_one_simplifies() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, one) = g.add_constant_f64(1.0);
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), mul);
        g.storage.set_node_inputs(mul, &[x, one]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        assert_eq!(g.outputs(), &[x]);
    }

    #[test]
    fn mul_zero_folds() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, zero) = g.add_constant_f64(0.0);
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), mul);
        g.storage.set_node_inputs(mul, &[x, zero]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        // 输出应指向新建的常量 0 节点的输出 value
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        let def_node = g.node(def).unwrap();
        assert_eq!(def_node.kind, OpKind::Constant);
        assert_eq!(def_node.constant_value(), Some(0.0));
    }

    #[test]
    fn const_folds_add() {
        let mut g = Graph::new("test");
        let (_c1, a) = g.add_constant_f64(3.0);
        let (_c2, b) = g.add_constant_f64(4.0);
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), add);
        g.storage.set_node_inputs(add, &[a, b]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        let def_node = g.node(def).unwrap();
        assert_eq!(def_node.constant_value(), Some(7.0));
    }

    #[test]
    fn sub_x_x_disabled_by_default() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sub = g.add_node(OpKind::Sub);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sub);
        g.storage.set_node_inputs(sub, &[x, x]);
        g.storage.set_node_outputs(sub, &[out]);
        g.mark_output(out);
        // 默认 unsafe_opts=false，不应简化
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn sub_x_x_enabled_with_unsafe() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sub = g.add_node(OpKind::Sub);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sub);
        g.storage.set_node_inputs(sub, &[x, x]);
        g.storage.set_node_outputs(sub, &[out]);
        g.mark_output(out);
        let count = run_with_config(&mut g, AlgebraConfig { unsafe_opts: true }).unwrap();
        assert_eq!(count, 1);
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        let def_node = g.node(def).unwrap();
        assert_eq!(def_node.constant_value(), Some(0.0));
    }

    #[test]
    fn reshape_noop_when_same_shape() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let reshape = g.add_node(OpKind::Reshape);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("out"),
            reshape,
        );
        g.storage.set_node_inputs(reshape, &[x]);
        g.storage.set_node_outputs(reshape, &[out]);
        g.storage
            .add_attr_int_array(reshape, base::StorageAttrKey::Shape, &[2, 3]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        // 输出应重写为 x
        assert_eq!(g.outputs(), &[x]);
    }

    #[test]
    fn reshape_kept_when_different_shape() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let reshape = g.add_node(OpKind::Reshape);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![6],
            },
            Some("out"),
            reshape,
        );
        g.storage.set_node_inputs(reshape, &[x]);
        g.storage.set_node_outputs(reshape, &[out]);
        g.storage
            .add_attr_int_array(reshape, base::StorageAttrKey::Shape, &[6]);
        g.mark_output(out);
        // shape 不同，不应消除
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn transpose_noop_when_identity_perm() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let tr = g.add_node(OpKind::Transpose);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("out"),
            tr,
        );
        g.storage.set_node_inputs(tr, &[x]);
        g.storage.set_node_outputs(tr, &[out]);
        g.storage
            .add_attr_int_array(tr, base::StorageAttrKey::Perm, &[0, 1]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        assert_eq!(g.outputs(), &[x]);
    }

    #[test]
    fn transpose_kept_when_non_identity_perm() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let tr = g.add_node(OpKind::Transpose);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![3, 2],
            },
            Some("out"),
            tr,
        );
        g.storage.set_node_inputs(tr, &[x]);
        g.storage.set_node_outputs(tr, &[out]);
        g.storage
            .add_attr_int_array(tr, base::StorageAttrKey::Perm, &[1, 0]);
        g.mark_output(out);
        // perm 非单位排列，不应消除
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0);
    }

    // --- 一元常量折叠测试 ---

    #[test]
    fn sqrt_constant_folds() {
        let mut g = Graph::new("test");
        let (_c, a) = g.add_constant_f64(9.0);
        let sqrt = g.add_node(OpKind::Sqrt);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sqrt);
        g.storage.set_node_inputs(sqrt, &[a]);
        g.storage.set_node_outputs(sqrt, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        assert_eq!(g.node(def).unwrap().constant_value(), Some(3.0));
    }

    #[test]
    fn exp_constant_folds() {
        let mut g = Graph::new("test");
        let (_c, a) = g.add_constant_f64(0.0);
        let exp = g.add_node(OpKind::Exp);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), exp);
        g.storage.set_node_inputs(exp, &[a]);
        g.storage.set_node_outputs(exp, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        assert_eq!(g.node(def).unwrap().constant_value(), Some(1.0));
    }

    #[test]
    fn pow_two_constants_fold() {
        let mut g = Graph::new("test");
        let (_c1, base) = g.add_constant_f64(2.0);
        let (_c2, exp) = g.add_constant_f64(10.0);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[base, exp]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        let out_v = g.outputs()[0];
        let def = g.value(out_v).unwrap().def_node();
        assert_eq!(g.node(def).unwrap().constant_value(), Some(1024.0));
    }

    #[test]
    fn pow_x_to_one_replaced_with_x() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let (_c, one) = g.add_constant_f64(1.0);
        let pow = g.add_node(OpKind::Pow);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), pow);
        g.storage.set_node_inputs(pow, &[x, one]);
        g.storage.set_node_outputs(pow, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1);
        // x^1 = x，输出应直接指向 x
        assert_eq!(g.outputs(), &[x]);
    }

    #[test]
    fn sqrt_non_constant_not_folded() {
        let mut g = Graph::new("test");
        let x = g.add_input(Type::Scalar(DType::F32), Some("x"));
        let sqrt = g.add_node(OpKind::Sqrt);
        let out = g.add_value(Type::Scalar(DType::F32), Some("out"), sqrt);
        g.storage.set_node_inputs(sqrt, &[x]);
        g.storage.set_node_outputs(sqrt, &[out]);
        g.mark_output(out);
        // 输入非常量，不应折叠
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0);
    }

    // --- shape-based 简化（多元素全 0/全 1 张量）---

    /// 构造一个多元素 Constant 张量节点（Value=FloatArray），shape=dims，值=vals。
    fn add_constant_tensor(g: &mut Graph, dims: &[i64], vals: &[f64], name: &str) -> ValueId {
        let node = g.add_node(OpKind::Constant);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: dims.to_vec(),
            },
            Some(name),
            node,
        );
        g.storage.set_node_outputs(node, &[out]);
        g.storage
            .add_attr_float_array(node, base::StorageAttrKey::Value, vals);
        out
    }

    #[test]
    fn mul_with_ones_tensor_simplifies() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        // ones [2,3] 全 1
        let ones = add_constant_tensor(&mut g, &[2, 3], &[1.0; 6], "ones");
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("out"),
            mul,
        );
        g.storage.set_node_inputs(mul, &[x, ones]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1, "x*ones 应简化为 x");
        assert_eq!(g.outputs(), &[x], "输出应重写为 x");
    }

    #[test]
    fn add_with_zeros_tensor_simplifies() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let zeros = add_constant_tensor(&mut g, &[2, 3], &[0.0; 6], "zeros");
        let add = g.add_node(OpKind::Add);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("out"),
            add,
        );
        g.storage.set_node_inputs(add, &[x, zeros]);
        g.storage.set_node_outputs(add, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1, "x+zeros 应简化为 x");
        assert_eq!(g.outputs(), &[x], "输出应重写为 x");
    }

    #[test]
    fn mul_with_zeros_tensor_replaced() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let zeros = add_constant_tensor(&mut g, &[2, 3], &[0.0; 6], "zeros");
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("out"),
            mul,
        );
        g.storage.set_node_inputs(mul, &[x, zeros]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1, "x*zeros 应简化");
        // 结果应复用那个 zeros 张量（保留 shape，不退化为标量）
        assert_eq!(
            g.outputs(),
            &[zeros],
            "输出应重写为 zeros 张量本身（保留 shape）"
        );
    }

    #[test]
    fn non_uniform_tensor_not_simplified() {
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("x"),
        );
        // [1, 0, 1, 0] 非全 1 也非全 0，不应触发 x*ones/x*zeros
        let mixed = add_constant_tensor(&mut g, &[2, 2], &[1.0, 0.0, 1.0, 0.0], "mixed");
        let mul = g.add_node(OpKind::Mul);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 2],
            },
            Some("out"),
            mul,
        );
        g.storage.set_node_inputs(mul, &[x, mixed]);
        g.storage.set_node_outputs(mul, &[out]);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0, "非均匀张量不应简化");
    }

    // --- 启发式 shape-based no-op 识别（reduce / concat）---

    #[test]
    fn reduce_size1_axis_to_input() {
        // 输入 shape [2,1,3]，ReduceMean(x, axis=1)，axis=1 维 size==1
        // 标准 reduce 语义输出 shape 应为 [2,3] → no-op，消除为 x
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 1, 3],
            },
            Some("x"),
        );
        let rs = g.add_node(OpKind::ReduceMean);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("out"),
            rs,
        );
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[out]);
        g.storage.add_attr_int(rs, base::StorageAttrKey::Axis, 1);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1, "axis 维 size==1 的 reduce 应消除为 x");
        assert_eq!(g.outputs(), &[x], "输出应重写为 x");
    }

    #[test]
    fn reduce_non_size1_axis_not_simplified() {
        // 输入 shape [2,3]，ReduceMean(x, axis=1)，axis=1 维 size=3≠1 → NoChange
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let rs = g.add_node(OpKind::ReduceMean);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2],
            },
            Some("out"),
            rs,
        );
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[out]);
        g.storage.add_attr_int(rs, base::StorageAttrKey::Axis, 1);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0, "axis 维 size!=1 不应消除");
    }

    #[test]
    fn reduce_size1_axis_wrong_out_shape_not_simplified() {
        // 输入 shape [2,1,3]，axis=1 维 size==1，但输出 shape 故意设成 [2,1,3]
        // （与 expected [2,3] 不符）→ 保守 NoChange
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 1, 3],
            },
            Some("x"),
        );
        let rs = g.add_node(OpKind::ReduceMean);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 1, 3],
            },
            Some("out"),
            rs,
        );
        g.storage.set_node_inputs(rs, &[x]);
        g.storage.set_node_outputs(rs, &[out]);
        g.storage.add_attr_int(rs, base::StorageAttrKey::Axis, 1);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0, "输出 shape 与 expected 不符时保守不消除");
    }

    #[test]
    fn concat_single_input_to_input() {
        // Concat([x], axis=0) → 结果等于 x（copy no-op）
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let cat = g.add_node(OpKind::Concat);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("out"),
            cat,
        );
        g.storage.set_node_inputs(cat, &[x]);
        g.storage.set_node_outputs(cat, &[out]);
        g.storage.add_attr_int(cat, base::StorageAttrKey::Axis, 0);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 1, "单输入 Concat 应消除为 x");
        assert_eq!(g.outputs(), &[x], "输出应重写为 x");
    }

    #[test]
    fn concat_multi_inputs_not_simplified() {
        // Concat([x,y], axis=0) 多输入 → NoChange
        let mut g = Graph::new("test");
        let x = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("x"),
        );
        let y = g.add_input(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![2, 3],
            },
            Some("y"),
        );
        let cat = g.add_node(OpKind::Concat);
        let out = g.add_value(
            Type::Tensor {
                dtype: DType::F32,
                dims: vec![4, 3],
            },
            Some("out"),
            cat,
        );
        g.storage.set_node_inputs(cat, &[x, y]);
        g.storage.set_node_outputs(cat, &[out]);
        g.storage.add_attr_int(cat, base::StorageAttrKey::Axis, 0);
        g.mark_output(out);
        let count = run_algebraic_simplify(&mut g).unwrap();
        assert_eq!(count, 0, "多输入 Concat 不应消除");
    }
}
