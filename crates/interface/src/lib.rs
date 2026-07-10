//! interface — 唯一公开功能 API：compile

use base::Result;
use isel::Instruction;

/// 编译输入
#[derive(Debug, Clone)]
pub enum Input {
    Onnx(Vec<u8>),
    Dsl(String),
    Pt(Vec<u8>),
}

/// 编译输出
#[derive(Debug, Clone)]
pub struct Output {
    pub target: String,
    pub instructions: Vec<Instruction>,
    pub debug: Option<String>,
}

/// 编译入口
pub fn compile(input: Input, config: Config) -> Result<Output> {
    let mut debug = String::new();

    // 1. 前端
    let mut graph = match &input {
        Input::Onnx(bytes) => frontend::parse_onnx(bytes)?,
        Input::Dsl(src) => frontend::dsl::parse(src)?,
        Input::Pt(bytes) => frontend::pt::parse(bytes)?,
    };

    if config.dump_ir {
        debug.push_str("// === 前端输出 ===\n");
        debug.push_str(&common::dump_graph(&graph));
    }

    // 2. 架构无关优化（三阶段：拆细→重排→融合）
    let mut pm = optimizer::PassManager::default_for(config.opt_level, config.target);
    pm.run(&mut graph)?;

    if config.dump_ir {
        debug.push_str("\n// === 优化后 ===\n");
        debug.push_str(&common::dump_graph(&graph));
    }

    // 3. Lowering
    let arch_graph = arch::lower(&graph, config.target)?;

    if config.dump_ir {
        debug.push_str(&format!(
            "\n// === Lowering 后 ({} ops) ===\n",
            arch_graph.len()
        ));
        for (i, op) in arch_graph.ops.iter().enumerate() {
            debug.push_str(&format!("  [{}] {:?}\n", i, op));
        }
    }

    // 4. 指令选择
    let instructions = isel::select(&arch_graph)?;

    if config.dump_ir {
        debug.push_str(&format!(
            "\n// === 最终指令 ({} 条) ===\n",
            instructions.len()
        ));
        for (i, ins) in instructions.iter().enumerate() {
            debug.push_str(&format!("  [{}] {} {}\n", i, ins.op, ins.args.join(" ")));
        }
    }

    Ok(Output {
        target: format!("{:?}", config.target).to_lowercase(),
        instructions,
        debug: if config.dump_ir { Some(debug) } else { None },
    })
}

// 重导出 common::Config 让外部用
pub use common::Config;
pub use common::{OptLevel, Target};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_empty_onnx() {
        let cfg = Config {
            target: Target::Cuda,
            opt_level: OptLevel::O2,
            dump_ir: true,
            trace_isel: false,
            algebra_unsafe_opts: false,
        };
        let out = compile(Input::Onnx(vec![]), cfg).unwrap();
        assert_eq!(out.target, "cuda");
        // 空图经 DCE 后 Placeholder 被删（无输出=死代码），指令为空
        assert!(out.instructions.is_empty());
        assert!(out.debug.is_some());
    }

    // --- 端到端：frontend ONNX 属性 → decompose ---

    fn wv(buf: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            buf.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        buf.push(v as u8);
    }
    fn wtag(buf: &mut Vec<u8>, field: u32, wt: u8) {
        wv(buf, ((field as u64) << 3) | (wt as u64));
    }
    fn wstr(buf: &mut Vec<u8>, field: u32, s: &str) {
        wtag(buf, field, 2);
        wv(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }
    fn wlen(buf: &mut Vec<u8>, field: u32, inner: &[u8]) {
        wtag(buf, field, 2);
        wv(buf, inner.len() as u64);
        buf.extend_from_slice(inner);
    }

    /// 构造 ONNX: LayerNormalization(x, gamma, beta, epsilon=eps) -> y
    /// LayerNorm 期望 3 个输入（x/scale/bias），decompose 才会拆。
    fn build_layernorm_onnx(eps: f32) -> Vec<u8> {
        // AttributeProto: name="epsilon"(1) + f=eps(4, FIXED32)
        let attr = {
            let mut a = Vec::new();
            wstr(&mut a, 1, "epsilon");
            wtag(&mut a, 4, 5);
            a.extend_from_slice(&eps.to_le_bytes());
            a
        };
        // NodeProto: inputs x/gamma/beta(1) + output y(2) + op_type(3) + attribute(5)
        let node = {
            let mut n = Vec::new();
            wstr(&mut n, 1, "x");
            wstr(&mut n, 1, "gamma");
            wstr(&mut n, 1, "beta");
            wstr(&mut n, 2, "y");
            wstr(&mut n, 3, "LayerNormalization");
            wlen(&mut n, 5, &attr);
            n
        };
        let vi = |name: &str| {
            let mut v = Vec::new();
            wstr(&mut v, 1, name);
            v
        };
        let graph = {
            let mut g = Vec::new();
            wlen(&mut g, 1, &node);
            wlen(&mut g, 11, &vi("x"));
            wlen(&mut g, 11, &vi("gamma"));
            wlen(&mut g, 11, &vi("beta"));
            wlen(&mut g, 12, &vi("y"));
            g
        };
        let mut buf = Vec::new();
        wlen(&mut buf, 7, &graph);
        buf
    }

    #[test]
    fn frontend_onnx_attributes_drive_decompose() {
        let bytes = build_layernorm_onnx(1e-3);
        // 1. 前端解析
        let mut graph = frontend::parse_onnx(&bytes).unwrap();
        // 前端应把 epsilon 写到 LayerNorm 节点的 Epsilon attr
        let ln_id = graph
            .node_ids()
            .find(|&id| {
                graph
                    .node(id)
                    .map(|n| n.kind == base::OpKind::LayerNorm)
                    .unwrap_or(false)
            })
            .expect("应有 LayerNorm 节点");
        let mut eps: Option<f64> = None;
        for e in graph.node(ln_id).unwrap().attrs() {
            if e.key == base::StorageAttrKey::Epsilon as u8
                && e.tag == base::storage::AttrTag::Float as u8
            {
                eps = Some(graph.node(ln_id).unwrap().storage.attr_float(e));
            }
        }
        let eps = eps.expect("前端应写入 Epsilon attr");
        assert!((eps - 1e-3).abs() < 1e-9, "epsilon 应为 1e-3，实际 {eps}");

        // 2. 单独跑 decompose（隔离 frontend→decompose 流，不被 fusion 干扰）
        let results = optimizer::decompose::run_decompose(&mut graph).unwrap();
        assert_eq!(results.len(), 1, "应拆分 1 个 LayerNorm");
        assert!(
            results[0].expanded.len() >= 8,
            "LayerNorm 应拆出至少 8 个原语节点，实际 {}",
            results[0].expanded.len()
        );

        // 3. 原 LayerNorm 节点应被删除
        let has_layernorm = graph.node_ids().any(|id| {
            graph
                .node(id)
                .map(|n| n.kind == base::OpKind::LayerNorm)
                .unwrap_or(false)
        });
        assert!(!has_layernorm, "LayerNorm 应被 decompose 拆掉");

        // 4. 应出现 decompose 产生的原语（ReduceMean/Sub/Sqrt/Div/Mul/Add）
        let kinds: std::collections::HashSet<_> = graph
            .node_ids()
            .filter_map(|id| graph.node(id).ok().map(|n| n.kind))
            .collect();
        assert!(kinds.contains(&base::OpKind::ReduceMean), "应有 ReduceMean");
        assert!(kinds.contains(&base::OpKind::Sub), "应有 Sub");
        assert!(kinds.contains(&base::OpKind::Sqrt), "应有 Sqrt");
        assert!(kinds.contains(&base::OpKind::Div), "应有 Div");
        assert!(kinds.contains(&base::OpKind::Mul), "应有 Mul");
        assert!(kinds.contains(&base::OpKind::Add), "应有 Add");
    }

    // --- 端到端：LayerNorm decompose 产生 Div(1,Sqrt) → float_opts 融合 Rsqrt ---

    /// LayerNorm decompose 会产生 `Div(1.0, Sqrt(var+ε))`，正是 `1/sqrt` 模式。
    /// float_opts 的 FastInvSqrt 应把它融合成 Rsqrt，且 Rsqrt 全链路通：
    /// lowering 发 "rsqrt" kernel、isel 选 "rsqrt" 指令。证明 IEEE754 浮点结构
    /// 优化在全 pipeline 生效，新 Rsqrt op 不报"未覆盖"。
    ///
    /// 跑完整 O2 pipeline（含 fusion）：fusion 产 Fused 节点（非 Custom），
    /// lowering 已覆盖 Fused → "fused" kernel、isel 选 "fused" 指令，不崩。
    #[test]
    fn layernorm_decompose_then_fast_inv_sqrt_fuses_to_rsqrt() {
        let bytes = build_layernorm_onnx(1e-3);
        let mut graph = frontend::parse_onnx(&bytes).unwrap();
        // 跑完整优化 pipeline O2（decompose → ... → float_opts → cse → dce →
        // fusion → dce）。fusion 产 Fused 节点，lowering 已覆盖不崩
        let mut pm = optimizer::PassManager::default_for(OptLevel::O2, Target::Cuda);
        pm.run(&mut graph).unwrap();

        // float_opts 应把 decompose 产生的 Div(1,Sqrt) 融合成 Rsqrt
        let has_rsqrt = graph
            .node_ids()
            .any(|id| graph.node(id).map(|n| n.kind == base::OpKind::Rsqrt).unwrap_or(false));
        assert!(
            has_rsqrt,
            "LayerNorm 经 decompose + float_opts 后应出现 Rsqrt 节点（1/sqrt 融合）"
        );

        // 全链路通：lowering 不报"未覆盖"（Fused + Rsqrt + 各原语都覆盖），isel 能选指令
        let arch_graph = arch::lower(&graph, Target::Cuda).unwrap();
        let instrs = isel::select(&arch_graph).unwrap();
        let has_rsqrt_instr = instrs.iter().any(|i| i.op == "rsqrt");
        assert!(has_rsqrt_instr, "isel 应为 Rsqrt 选出 'rsqrt' 指令");
        // 若 fusion 产生了 Fused 节点，应选出 "fused" 指令（不崩在 lowering）
        let has_fused_instr = instrs.iter().any(|i| i.op == "fused");
        let has_fused_node = graph
            .node_ids()
            .any(|id| graph.node(id).map(|n| n.kind == base::OpKind::Fused).unwrap_or(false));
        assert_eq!(
            has_fused_instr, has_fused_node,
            "Fused 节点与 'fused' 指令应一致出现"
        );
    }
}
