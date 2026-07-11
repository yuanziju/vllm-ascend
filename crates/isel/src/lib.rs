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
//! - `emit` 的参数是 lisp 表达式，第一个求值结果作指令 op 名，其余（"r0"/"r1" 等占位符）
//!   解析后丢弃——真实 operand 由 select_with_rules 从 ArchOp 的 ValueId 分配 VReg 填充
//!
//! 例：`(rule (when (= op "add")) (emit "fadd" "r0" "r1"))`
//! 例：`(rule (when (and (= op "mma") (= target "cuda"))) (emit "wgmma" "a" "b"))`

use std::collections::HashMap;

use arch::{ArchGraph, ArchOp};
use base::{NeutronError, Result, ValueId};
use lisp::{parse, Interp, Val};

/// 虚拟寄存器 ID（u32 索引，由 isel 分配，待 regalloc 映射到物理寄存器）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(pub u32);

/// 指令 operand：虚拟寄存器或立即数
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operand {
    /// 虚拟寄存器（def-use chain 节点，待 regalloc 分配物理寄存器）
    VReg(VReg),
    /// 立即数（常量，不占寄存器）
    Imm(f64),
}

/// 最终指令（带虚拟寄存器 operand，支撑寄存器分配的 def-use 追踪）
#[derive(Debug, Clone)]
pub struct Instruction {
    pub op: String,
    /// 输入 operand 列表（VReg 或 Imm）
    pub inputs: Vec<Operand>,
    /// 输出 VReg 列表（指令 def 的虚拟寄存器）
    pub outputs: Vec<VReg>,
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
        r#"(rule (when (= op "rsqrt"))  (emit "rsqrt" "x"))"#,
        r#"(rule (when (= op "reciprocal")) (emit "reciprocal" "x"))"#,
        r#"(rule (when (= op "abs"))    (emit "abs"   "x"))"#,
        r#"(rule (when (= op "log"))    (emit "log"   "x"))"#,
        r#"(rule (when (= op "exp"))    (emit "exp"   "x"))"#,
        r#"(rule (when (= op "pow"))    (emit "pow"   "x" "y"))"#,
        // reduce
        r#"(rule (when (= op "reduce_sum"))  (emit "rsum"  "x" "axis"))"#,
        r#"(rule (when (= op "reduce_mean")) (emit "rmean" "x" "axis"))"#,
        r#"(rule (when (= op "reduce_max"))  (emit "rmax"  "x" "axis"))"#,
        // 数据移动（无 FLOPs，仅布局/形状调整）
        r#"(rule (when (= op "reshape"))   (emit "reshape" "x"))"#,
        r#"(rule (when (= op "transpose")) (emit "transpose" "x"))"#,
        r#"(rule (when (= op "concat"))    (emit "concat" "x"))"#,
        r#"(rule (when (= op "slice"))     (emit "slice" "x"))"#,
        r#"(rule (when (= op "pool"))      (emit "pool" "x"))"#,
        // 复合（未拆细时直发）
        r#"(rule (when (= op "softmax"))    (emit "sm"    "x"))"#,
        r#"(rule (when (= op "layer_norm")) (emit "ln"    "x" "g" "b"))"#,
        r#"(rule (when (= op "conv"))       (emit "conv"  "x" "w"))"#,
        // 访存
        r#"(rule (when (= op "load"))   (emit "load"  "addr"))"#,
        r#"(rule (when (= op "store"))  (emit "store" "addr" "v"))"#,
        r#"(rule (when (= op "const"))  (emit "const" "imm"))"#,
        // 融合产物（fuse pass 产，attr 记 op 序列）；未知 ONNX 算子（frontend 产）
        r#"(rule (when (= op "fused"))  (emit "fused" "x"))"#,
        r#"(rule (when (= op "custom")) (emit "custom" "x"))"#,
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

/// 从规则文本源加载多条规则。源里每条规则用 `(rule ...)` 表示，
/// 可有多条，之间空白分隔。注释以 `;` 开头到行尾。
/// 例：
/// ```text
/// ; add 规则
/// (rule (when (= op "add")) (emit "fadd" "r0" "r1"))
/// (rule (when (= op "mul")) (emit "fmul" "r0" "r1"))
/// ```
pub fn load_rules_from_src(src: &str) -> Result<Vec<Rule>> {
    // 顶层可能有多个 (rule ...)。用括号配平切分。
    let mut rules = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // 跳过空白和注释
        let b = bytes[i];
        if b == b';' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b != b'(' {
            return Err(NeutronError::Isel(format!(
                "规则源第 {} 字节处期望 '('，得到 {:?}",
                i, b as char
            )));
        }
        // 配平括号，截取一个完整 S-expr
        let start = i;
        let mut depth = 0i32;
        while i < bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                b';' => {
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
        if depth != 0 {
            return Err(NeutronError::Isel("规则源括号不配平".into()));
        }
        let expr = &src[start..=i];
        i += 1; // 跳过 ')'
        let rule = parse_rule(expr)?;
        rules.push(rule);
    }
    Ok(rules)
}

/// 从文件路径加载规则集（热加载，不重编译）。文件格式同 [`load_rules_from_src`]。
pub fn load_rules_from_file(path: &str) -> Result<Vec<Rule>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| NeutronError::Isel(format!("读取规则文件 {} 失败: {}", path, e)))?;
    load_rules_from_src(&content)
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
        // emit 的 args（"r0"/"r1" 等占位字符串）解析后丢弃——真实 operand 由
        // select_with_rules 从 ArchOp 的 ValueId 分配 VReg 填充（寄存器分配
        // 需要可追踪的 def-use chain，而非字面占位符）
        return Ok(Some(Instruction {
            op: instr_op,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }));
    }
    Ok(None)
}

/// 从 ArchGraph 选择指令（用 lisp 规则驱动）
pub fn select(arch_graph: &ArchGraph) -> Result<Vec<Instruction>> {
    select_with_rules(arch_graph, &default_rules())
}

/// 分配/复用虚拟寄存器：ValueId 已在 map 中则复用（同一 def-use chain 节点），
/// 否则分配新 VReg。SSA 下每个 ValueId 恰被定义一次，故 output ValueId 总是新 def。
fn alloc_vreg(vid: ValueId, map: &mut HashMap<ValueId, VReg>, next: &mut u32) -> VReg {
    if let Some(&v) = map.get(&vid) {
        v
    } else {
        let v = VReg(*next);
        *next += 1;
        map.insert(vid, v);
        v
    }
}

/// 用自定义规则集选择指令，维护 ValueId→VReg 映射
pub fn select_with_rules(arch_graph: &ArchGraph, rules: &[Rule]) -> Result<Vec<Instruction>> {
    let target = format!("{:?}", arch_graph.target).to_lowercase();
    let mut instrs = Vec::new();
    let mut vreg_map: HashMap<ValueId, VReg> = HashMap::new();
    let mut next_vreg: u32 = 0;
    for (i, op) in arch_graph.ops.iter().enumerate() {
        // 取 op 名 + 输入/输出 ValueId 列表（ArchOp 已携带，前置缺口1）
        let (op_name, inputs_vids, outputs_vids): (&str, &[ValueId], &[ValueId]) = match op {
            ArchOp::KernelCall {
                name,
                inputs,
                outputs,
            } => (name.as_str(), inputs, outputs),
            ArchOp::Load { addr, dst } => ("load", &[*addr][..], &[*dst][..]),
            ArchOp::Store { addr, src } => ("store", &[*addr, *src][..], &[][..]),
        };
        match select_one(op_name, i, &target, rules)? {
            Some(mut ins) => {
                // 用 ArchOp 的 ValueId 填充 Instruction 的 operand：
                // inputs 复用已有 VReg（def-use chain），outputs 分配新 VReg（新 def）
                ins.inputs = inputs_vids
                    .iter()
                    .map(|&v| Operand::VReg(alloc_vreg(v, &mut vreg_map, &mut next_vreg)))
                    .collect();
                ins.outputs = outputs_vids
                    .iter()
                    .map(|&v| alloc_vreg(v, &mut vreg_map, &mut next_vreg))
                    .collect();
                instrs.push(ins);
            }
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
        let ag = make_arch(
            Target::Cuda,
            vec![ArchOp::KernelCall {
                name: "add".into(),
                inputs: vec![],
                outputs: vec![],
            }],
        );
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].op, "fadd");
        // ArchOp 无 inputs/outputs → Instruction 的 inputs/outputs 为空
        // （真实 operand 由 select_with_rules 从 ValueId 填充）
        assert!(instrs[0].inputs.is_empty());
        assert!(instrs[0].outputs.is_empty());
    }

    #[test]
    fn selects_mma_for_cuda() {
        let ag = make_arch(
            Target::Cuda,
            vec![ArchOp::KernelCall {
                name: "mma".into(),
                inputs: vec![],
                outputs: vec![],
            }],
        );
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs[0].op, "mma");
        assert!(instrs[0].inputs.is_empty());
        assert!(instrs[0].outputs.is_empty());
    }

    #[test]
    fn vreg_mapping_tracks_def_use() {
        // 两条 add 指令的 def-use chain：
        //   op0: add(v10, v11) -> v12   （v10/v11 是 graph 输入，v12 新 def）
        //   op1: add(v12, v10) -> v13   （v12 复用 op0 的输出 VReg，v10 复用，v13 新 def）
        // VReg 分配应为：v10→VReg(0), v11→VReg(1), v12→VReg(2), v13→VReg(3)
        // op1 的第一个 input 应复用 op0 的 output VReg(2)——这正是寄存器分配
        // 追踪 def-use chain 所需的真实 operand 关系（旧 args 字面 "r0"/"r1" 无法表达）
        let ag = make_arch(
            Target::Cuda,
            vec![
                ArchOp::KernelCall {
                    name: "add".into(),
                    inputs: vec![10, 11],
                    outputs: vec![12],
                },
                ArchOp::KernelCall {
                    name: "add".into(),
                    inputs: vec![12, 10],
                    outputs: vec![13],
                },
            ],
        );
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs.len(), 2);

        // op0：inputs=[VReg(0), VReg(1)]，outputs=[VReg(2)]
        assert_eq!(instrs[0].inputs.len(), 2);
        assert_eq!(instrs[0].outputs.len(), 1);
        let out0 = instrs[0].outputs[0];
        assert_eq!(out0, VReg(2));
        match instrs[0].inputs[0] {
            Operand::VReg(v) => assert_eq!(v, VReg(0)),
            Operand::Imm(_) => panic!("input 应是 VReg"),
        }

        // op1：inputs=[VReg(2)(复用 op0 输出), VReg(0)(复用 v10)]，outputs=[VReg(3)]
        assert_eq!(instrs[1].inputs.len(), 2);
        assert_eq!(instrs[1].outputs.len(), 1);
        assert_eq!(instrs[1].outputs[0], VReg(3));
        match instrs[1].inputs[0] {
            Operand::VReg(v) => {
                assert_eq!(v, out0, "v12 应复用 op0 的输出 VReg（def-use chain 衔接）")
            }
            Operand::Imm(_) => panic!("input 应是 VReg"),
        }
        match instrs[1].inputs[1] {
            Operand::VReg(v) => assert_eq!(v, VReg(0), "v10 应复用 op0 的 input VReg"),
            Operand::Imm(_) => panic!("input 应是 VReg"),
        }
    }

    #[test]
    fn load_store_vreg_operands() {
        // Load: input=[addr], output=[dst]；Store: input=[addr, src], output=[]
        let ag = make_arch(
            Target::Cpu,
            vec![
                ArchOp::Load { addr: 0, dst: 1 },
                ArchOp::Store { addr: 0, src: 1 },
            ],
        );
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs[0].op, "load");
        assert_eq!(instrs[1].op, "store");
        // load: 1 input (addr=0→VReg(0)), 1 output (dst=1→VReg(1))
        assert_eq!(instrs[0].inputs.len(), 1);
        assert_eq!(instrs[0].outputs.len(), 1);
        // store: 2 inputs (addr=0→复用 VReg(0), src=1→复用 VReg(1)), 0 outputs
        assert_eq!(instrs[1].inputs.len(), 2);
        assert!(instrs[1].outputs.is_empty());
        // store 的 addr 应复用 load 的 addr VReg（同一 ValueId 0）
        match (instrs[0].inputs[0], instrs[1].inputs[0]) {
            (Operand::VReg(a), Operand::VReg(b)) => assert_eq!(a, b, "addr VReg 应复用"),
            _ => panic!("addr 应是 VReg"),
        }
        // store 的 src 应复用 load 的 dst VReg（同一 ValueId 1）
        match (instrs[0].outputs[0], instrs[1].inputs[1]) {
            (load_dst, Operand::VReg(store_src)) => {
                assert_eq!(load_dst, store_src, "dst/src VReg 应复用（def-use）")
            }
            _ => panic!("dst/src 应是 VReg"),
        }
    }

    #[test]
    fn custom_rule_target_guard() {
        // 只在 cuda 上把 mma 发成 wgmma
        let rule = parse_rule(
            r#"(rule (when (and (= op "mma") (= target "cuda"))) (emit "wgmma" "a" "b"))"#,
        )
        .unwrap();
        // cuda 命中
        let ag_cuda = make_arch(
            Target::Cuda,
            vec![ArchOp::KernelCall {
                name: "mma".into(),
                inputs: vec![],
                outputs: vec![],
            }],
        );
        let ins = select_with_rules(&ag_cuda, std::slice::from_ref(&rule)).unwrap();
        assert_eq!(ins[0].op, "wgmma");
    }

    #[test]
    fn no_match_returns_error() {
        let ag = make_arch(
            Target::Cuda,
            vec![ArchOp::KernelCall {
                name: "unknown_op".into(),
                inputs: vec![],
                outputs: vec![],
            }],
        );
        let err = select(&ag).unwrap_err();
        assert!(matches!(err, NeutronError::Isel(_)));
    }

    #[test]
    fn load_store_selected() {
        let ag = make_arch(
            Target::Cpu,
            vec![
                ArchOp::Load { addr: 0, dst: 0 },
                ArchOp::Store { addr: 0, src: 0 },
            ],
        );
        let instrs = select(&ag).unwrap();
        assert_eq!(instrs[0].op, "load");
        assert_eq!(instrs[1].op, "store");
    }

    #[test]
    fn idx_bound_in_rule() {
        // 用 idx 生成参数：第 0 个发 "load_0"
        let rule = parse_rule(r#"(rule (when true) (emit (str "+" "load_" idx) "x"))"#).unwrap();
        let ag = make_arch(
            Target::Cuda,
            vec![ArchOp::KernelCall {
                name: "add".into(),
                inputs: vec![],
                outputs: vec![],
            }],
        );
        // 这条规则不依赖 op，true 恒命中
        let ins = select_with_rules(&ag, &[rule]).unwrap();
        assert!(ins[0].op.contains("load_"));
    }

    #[test]
    fn load_multiple_rules_from_src() {
        let src = r#"
            ; add 规则
            (rule (when (= op "add")) (emit "fadd" "r0" "r1"))
            ; mul 规则
            (rule (when (= op "mul")) (emit "fmul" "r0" "r1"))
        "#;
        let rules = load_rules_from_src(src).unwrap();
        assert_eq!(rules.len(), 2, "应加载 2 条规则");
        // 用加载的规则选 add
        let ag = make_arch(
            Target::Cuda,
            vec![ArchOp::KernelCall {
                name: "add".into(),
                inputs: vec![],
                outputs: vec![],
            }],
        );
        let ins = select_with_rules(&ag, &rules).unwrap();
        assert_eq!(ins[0].op, "fadd");
    }

    #[test]
    fn load_rules_from_file_works() {
        // 写临时规则文件
        let path = "/tmp/neutron_isel_test.rules";
        let content = r#"
            (rule (when (= op "custom_op")) (emit "my_instr" "a"))
        "#;
        std::fs::write(path, content).unwrap();
        let rules = load_rules_from_file(path).unwrap();
        assert_eq!(rules.len(), 1);
        let ag = make_arch(
            Target::Cuda,
            vec![ArchOp::KernelCall {
                name: "custom_op".into(),
                inputs: vec![],
                outputs: vec![],
            }],
        );
        let ins = select_with_rules(&ag, &rules).unwrap();
        assert_eq!(ins[0].op, "my_instr");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn unbalanced_parens_error() {
        let src = r#"(rule (when (= op "add") (emit "fadd"))"#; // 少一个 )
        let err = load_rules_from_src(src).unwrap_err();
        assert!(matches!(err, NeutronError::Isel(_)));
    }

    #[test]
    fn comments_stripped() {
        let src = "; 整行注释\n(rule (when true) (emit \"x\"))\n; 末尾注释";
        let rules = load_rules_from_src(src).unwrap();
        assert_eq!(rules.len(), 1);
    }
}
