//! isel — 指令选择（基于 lisp 规则）
//!
//! 设计哲学：用 S-expr 规则描述"op → 指令"映射，规则由 lisp 解释器求值，
//! 而非硬编码 match。规则可读、可扩展、可热加载（后期可从文件读）。
//!
//! 规则格式：
//! ```text
//! (rule (when <条件>) (emit <op> <arg0> <arg1> ...))
//! ```
//! - `when` 条件是 lisp 表达式，求值为非 nil/非 false 时触发
//! - 求值时绑定变量：`op`（op 名字符串）、`idx`（节点序号）、`target`（目标架构字符串）
//! - `emit` 的参数是 lisp 表达式，求值后拼成指令的 op + args
//!
//! 例：`(rule (when (= op "add")) (emit "fadd" "r0" "r1"))`
//! 例：`(rule (when (and (= op "mma") (= target "cuda"))) (emit "wgmma" "a" "b"))`

use arch::{ArchGraph, ArchOp};
use base::{NeutronError, Result};
use lisp::{parse, Interp, Val};

/// 最终指令
#[derive(Debug, Clone)]
pub struct Instruction {
    pub op: String,
    pub args: Vec<String>,
}

/// 一条 isel 规则
#[derive(Debug, Clone)]
pub struct Rule {
    /// when 条件表达式（已 parse）
    pub cond: Val,
    /// emit 表达式（已 parse 的 List）
    pub emit: Val,
    /// 源文本（调试用）
    pub src: String,
}

/// 默认规则集（覆盖 lowering 产出的 native kernel 名）
pub fn default_rules() -> Vec<Rule> {
    let srcs = [
        // 算术
        r#"(rule (when (= op "add"))    (emit "fadd"  "r0" "r1"))"#,
        r#"(rule (when (= op "sub"))    (emit "fsub"  "r0" "r1"))"#,
        r#"(rule (when (= op "mul"))    (emit "fmul"  "r0" "r1"))"#,
        r#"(rule (when (= op "div"))    (emit "fdiv"  "r0" "r1"))"#,
        // GEMM
        r#"(rule (when (= op "mma"))    (emit "mma"   "a" "b" "c"))"#,
        // 激活
        r#"(rule (when (= op "relu"))   (emit "relu"  "x"))"#,
        r#"(rule (when (= op "gelu"))   (emit "gelu"  "x"))"#,
        r#"(rule (when (= op "sigmoid"))(emit "sigm"  "x"))"#,
        r#"(rule (when (= op "tanh"))   (emit "tanh"  "x"))"#,
        // 超越函数
        r#"(rule (when (= op "sqrt"))   (emit "sqrt"  "x"))"#,
        r#"(rule (when (= op "exp"))    (emit "exp"   "x"))"#,
        r#"(rule (when (= op "pow"))    (emit "pow"   "x" "y"))"#,
        // reduce
        r#"(rule (when (= op "reduce_sum"))  (emit "rsum"  "x" "axis"))"#,
        r#"(rule (when (= op "reduce_mean")) (emit "rmean" "x" "axis"))"#,
        r#"(rule (when (= op "reduce_max"))  (emit "rmax"  "x" "axis"))"#,
        // 复合（未拆细时直发）
        r#"(rule (when (= op "softmax"))    (emit "sm"    "x"))"#,
        r#"(rule (when (= op "layer_norm")) (emit "ln"    "x" "g" "b"))"#,
        r#"(rule (when (= op "conv"))       (emit "conv"  "x" "w"))"#,
        // 访存
        r#"(rule (when (= op "load"))   (emit "load"  "addr"))"#,
        r#"(rule (when (= op "store"))  (emit "store" "addr" "v"))"#,
        r#"(rule (when (= op "const"))  (emit "const" "imm"))"#,
    ];
    srcs.iter().map(|s| parse_rule(s).unwrap()).collect()
}

/// 解析一条规则文本
pub fn parse_rule(src: &str) -> Result<Rule> {
    let v = parse(src).map_err(|e| NeutronError::Isel(format!("规则解析失败: {}", e)))?;
    let items = v
        .as_list()
        .ok_or_else(|| NeutronError::Isel("规则必须是 list".into()))?;
    if items.is_empty() || items[0].as_sym() != Some("rule") {
        return Err(NeutronError::Isel("规则必须以 (rule ...) 开头".into()));
    }
    let mut cond = Val::Nil;
    let mut emit = Val::Nil;
    for item in &items[1..] {
        let sub = item
            .as_list()
            .ok_or_else(|| NeutronError::Isel("rule 子句必须是 list".into()))?;
        if sub.is_empty() {
            continue;
        }
        match sub[0].as_sym() {
            Some("when") => {
                cond = sub
                    .get(1)
                    .cloned()
                    .ok_or_else(|| NeutronError::Isel("when 缺少条件表达式".into()))?;
            }
            Some("emit") => {
                emit = item.clone();
            }
            _ => {
                return Err(NeutronError::Isel(format!("未知 rule 子句: {:?}", sub[0])));
            }
        }
    }
    Ok(Rule {
        cond,
        emit,
        src: src.to_string(),
    })
}

/// 判定 lisp 值是否为"真"（非 nil、非 false）
fn is_true(v: &Val) -> bool {
    !matches!(v, Val::Nil | Val::Bool(false))
}

/// 把 lisp Val 转成字符串（emit 参数用）
fn val_to_str(v: &Val) -> String {
    match v {
        Val::Str(s) => s.clone(),
        Val::Sym(s) => s.clone(),
        Val::Int(i) => i.to_string(),
        Val::Float(f) => f.to_string(),
        Val::Bool(b) => b.to_string(),
        Val::Nil => "nil".to_string(),
        other => format!("{}", other),
    }
}

/// 对单个 ArchOp 应用规则集，返回匹配到的指令（取第一条命中规则）
fn select_one(
    op_name: &str,
    idx: usize,
    target: &str,
    rules: &[Rule],
) -> Result<Option<Instruction>> {
    for rule in rules {
        // 每条规则用独立 interp 绑定上下文
        let mut interp = Interp::new();
        interp
            .vars
            .insert("op".into(), Val::Str(op_name.to_string()));
        interp.vars.insert("idx".into(), Val::Int(idx as i64));
        interp
            .vars
            .insert("target".into(), Val::Str(target.to_string()));

        let cond_val = interp
            .eval(&rule.cond)
            .map_err(|e| NeutronError::Isel(format!("规则条件求值失败 [{}]: {}", rule.src, e)))?;
        if !is_true(&cond_val) {
            continue;
        }

        // 命中：求值 emit
        let emit_items = rule
            .emit
            .as_list()
            .ok_or_else(|| NeutronError::Isel("emit 必须是 list".into()))?;
        if emit_items.is_empty() || emit_items[0].as_sym() != Some("emit") {
            return Err(NeutronError::Isel("emit 子句格式错误".into()));
        }
        let mut parts = Vec::new();
        for arg in &emit_items[1..] {
            let v = interp.eval(arg).map_err(|e| {
                NeutronError::Isel(format!("emit 参数求值失败 [{}]: {}", rule.src, e))
            })?;
            parts.push(val_to_str(&v));
        }
        let instr_op = parts.remove(0);
        return Ok(Some(Instruction {
            op: instr_op,
            args: parts,
        }));
    }
    Ok(None)
}

/// 从 ArchGraph 选择指令（用 lisp 规则驱动）
pub fn select(arch_graph: &ArchGraph) -> Result<Vec<Instruction>> {
    select_with_rules(arch_graph, &default_rules())
}

/// 用自定义规则集选择指令
pub fn select_with_rules(arch_graph: &ArchGraph, rules: &[Rule]) -> Result<Vec<Instruction>> {
    let target = format!("{:?}", arch_graph.target).to_lowercase();
    let mut instrs = Vec::new();
    for (i, op) in arch_graph.ops.iter().enumerate() {
        let op_name = match op {
            ArchOp::KernelCall(name) => name.as_str(),
            ArchOp::Load => "load",
            ArchOp::Store => "store",
        };
        match select_one(op_name, i, &target, rules)? {
            Some(ins) => instrs.push(ins),
            None => {
                return Err(NeutronError::Isel(format!(
                    "无规则匹配 op: {:?}（idx {}）",
                    op, i
                )))
            }
        }
    }
    Ok(instrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arch::ArchOp;
    use common::Target;

    fn make_arch(target: Target, ops: Vec<ArchOp>) -> ArchGraph {
        let mut ag = ArchGraph::new(target);
        for o in ops {
            ag.add(o);
        }
        ag
    }

    #[test]
    fn parses_rule() {
        let r = parse_rule(r#"(rule (when (= op "add")) (emit "fadd" "r0" "r1"))"#).unwrap();
        // cond 应是 List (= op "add")
        assert!(r.cond.as_list().is_some());
        assert!(r.emit.as_list().is_some());
    }

    #[test]
    fn default_rules_cover_add() {
        let rules = default_rules();
        assert!(!rules.is_empty());
        // add 规则应存在
        assert!(rules.iter().any(|r| r.src.contains("\"add\"")));
    }

    #[test]
    fn selects_add_instruction() {
        let ag = make_arch(Target::Cuda, vec![ArchOp::KernelCall("add".into())]);
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].op, "fadd");
        assert_eq!(instrs[0].args, vec!["r0", "r1"]);
    }

    #[test]
    fn selects_mma_for_cuda() {
        let ag = make_arch(Target::Cuda, vec![ArchOp::KernelCall("mma".into())]);
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs[0].op, "mma");
        assert_eq!(instrs[0].args, vec!["a", "b", "c"]);
    }

    #[test]
    fn custom_rule_target_guard() {
        // 只在 cuda 上把 mma 发成 wgmma
        let rule = parse_rule(
            r#"(rule (when (and (= op "mma") (= target "cuda"))) (emit "wgmma" "a" "b"))"#,
        )
        .unwrap();
        // cuda 命中
        let ag_cuda = make_arch(Target::Cuda, vec![ArchOp::KernelCall("mma".into())]);
        let ins = select_with_rules(&ag_cuda, std::slice::from_ref(&rule)).unwrap();
        assert_eq!(ins[0].op, "wgmma");
    }

    #[test]
    fn no_match_returns_error() {
        let ag = make_arch(Target::Cuda, vec![ArchOp::KernelCall("unknown_op".into())]);
        let err = select(&ag).unwrap_err();
        assert!(matches!(err, NeutronError::Isel(_)));
    }

    #[test]
    fn load_store_selected() {
        let ag = make_arch(Target::Cpu, vec![ArchOp::Load, ArchOp::Store]);
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs[0].op, "load");
        assert_eq!(instrs[1].op, "store");
    }

    #[test]
    fn idx_bound_in_rule() {
        // 用 idx 生成参数：第 0 个发 "load_0"
        let rule = parse_rule(r#"(rule (when true) (emit (str "+" "load_" idx) "x"))"#).unwrap();
        let ag = make_arch(Target::Cuda, vec![ArchOp::KernelCall("add".into())]);
        // 这条规则不依赖 op，true 恒命中
        let ins = select_with_rules(&ag, &[rule]).unwrap();
        assert!(ins[0].op.contains("load_"));
    }
}
