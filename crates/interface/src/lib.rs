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
        debug.push_str(&format!("\n// === Lowering 后 ({} ops) ===\n", arch_graph.len()));
        for (i, op) in arch_graph.ops.iter().enumerate() {
            debug.push_str(&format!("  [{}] {:?}\n", i, op));
        }
    }

    // 4. 指令选择
    let instructions = isel::select(&arch_graph)?;

    if config.dump_ir {
        debug.push_str(&format!("\n// === 最终指令 ({} 条) ===\n", instructions.len()));
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
        };
        let out = compile(Input::Onnx(vec![]), cfg).unwrap();
        assert_eq!(out.target, "cuda");
        // 空图经 DCE 后 Placeholder 被删（无输出=死代码），指令为空
        assert!(out.instructions.is_empty());
        assert!(out.debug.is_some());
    }
}
