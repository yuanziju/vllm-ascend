# AGENTS.md

Guidance for AI coding agents working in the **Neutron** repository.

> 这是 Neutron 项目（Rust ML 编译器）的 agent 指南。前任 agent 因 P0 级事故销毁过项目，本文件含**防 P0 安全规则**与 **Continuity Log（进度遗言）**，每轮工作前必读。

## Project Overview

Neutron 是一个用 Rust 编写的 ML 编译器，把高层模型（ONNX / 自研 DSL / PyTorch）经架构无关优化 + 架构相关 lowering + 指令选择，编译成目标后端（CUDA / Ascend NPU / CPU）的指令序列。

- License: Apache-2.0
- Language: Rust (edition 2021, MSRV 1.75)
- 9-crate workspace（**无 `neutron-` 前缀**，crate 名即职责名）

## 设计哲学（与用户深度讨论后敲定，不可擅自变更）

### 架构无关 IR
- **范式**：MLIR 风格（一切皆 op + region），统一框架、可嵌套表达控制流
- **层次**：分层渐进 lowering（HLO → LLO）
- **副作用**：纯函数式 SSA（无副作用，重排自由）
- **值流转**：tagged value ID（值 ID 编码类型 tag，省查表；`TypeTag` 高位 0x80 = tensor）
- **类型**：静态类型 + shape 进入类型系统（依赖类型）
- **存储**：连续 packed buffer + unsafe + Safe 包装。上层 `Graph`（Safe API），下层委托 `raw::RawGraph`（巨量 unsafe 构建丑陋但高效的王国）。`#[repr(C)]` 定长头 + 连续 `Vec` 池，**ID = 偏移量，O(1) 访问**

### 优化哲学（用户多次强调，核心约束）
- **不要模式匹配**：不硬编码 `MatMul+Add→Linear` 这类贪心模式
- **用简单代数规则**：`x+0=x`, `x*1=x`, `x-x=0`（保守，NaN 风险，默认不启用 `x-x`）
- **浮点结构优化**：针对 IEEE754 位级 trick（类 Quake III `fast inverse sqrt` 那句悠扬婉转的注释；Flash Attention online-softmax 式重排）
- **IO 同样性**：CSE 公共子表达式消除
- **启发式优化器 + cost model**：cost model 现在就做（FLOPs + memory access + launch overhead，CUDA/NPU/CPU 不同系数）
- **三阶段 pipeline**：拆细（一对多 decompose）→ 重排（algebra + float + CSE）→ 融合（多对一 fuse，带 cost model）
- **规则用函数实现**，后期再抽象宏（先函数，后期抽象宏）

## Repository Layout

```
Cargo.toml              # workspace 根，9 crate
crates/
  base/                 # IR 核心：raw.rs (packed buffer unsafe) + lib.rs (Safe Graph/NodeView/ValueView)
                        #   NeutronError, NodeId/ValueId, DType/TypeTag/Type, OpKind, Attr, Pass/PassContext/Visitor
  common/               # Target(Cuda/Npu/Cpu), OptLevel, IdGen, Arena, Config, dump_graph
  frontend/             # onnx.rs / dsl.rs / pt.rs 前端解析（当前占位）
  optimizer/            # 三阶段 pipeline 入口 + passes(DCE/Verify) + algebra + float_opts + cse + decompose + fuse + cost_model
  arch/                 # ArchGraph + lower() 1:1 op→native kernel + cuda.rs/npu.rs 设备描述
  lisp/                 # S-expr 解释器（parser + interp），用于 isel 规则
  isel/                 # 指令选择 select()
  interface/            # 唯一公开 API compile()，串联 frontend→optimizer→arch→isel
  cli/                  # neutron 二进制，--target/--opt/--dump
```

## Build & Test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
# CLI 端到端
cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump
```

## 防 P0 安全规则（血泪教训，必须遵守）

1. **工作前先建新分支**：从 `main`（或当前主干）`git checkout -b feat/<name>`。在新分支上工作，崩了不影响主干。
2. **没崩就合并**：分支工作完成且通过回归验证（build + test + clippy + fmt），合并回主干；下一轮工作继续用主干。
3. **频繁提交**：每完成一个独立单元（一个 crate / 一个 pass / 一组修复）立即 commit，不要攒一堆。
4. **DCO sign-off**：每个 commit 用 `git commit -s` 加 `Signed-off-by`。
5. **严格控制子代理的 git 使用**：子代理只读 / 只写文件，**禁止子代理直接 git commit / merge / push**，所有 git 操作由主 agent 串行执行。历史 P0 就是子代理并行提交时序冲突导致。
6. **不要 `git add -A` / `git add .`**：用具体路径，避免误加敏感文件或 vllm-ascend 残留。
7. **根 Cargo.toml 必须尽早提交**：它是 workspace 入口，丢失会导致整个项目无法编译。

## Conventions

- **Imports**：`use` 按 std → external → crate 分组
- **错误**：统一用 `thiserror`，`base::NeutronError` + `base::Result<T>`
- **注释**：中文注释，技术术语保留英文（packed buffer / SSA / MLIR / cost model / CSE / IEEE754 等）
- **文件头**：每个 `.rs` 顶部用 `//!` 模块注释说明职责
- **unsafe**：集中在 `base/src/raw.rs`，上层只暴露 Safe API

## 待办（vllm-ascend 残留清理）

> 本仓库 git 历史从 vllm-ascend fork 而来，`main` 上还残留大量 vllm-ascend 文件（`vllm_ascend/`、`csrc/`、`docs/`、`tests/`、`benchmarks/`、`examples/`、`tools/`、`.github/`、`Dockerfile*`、`setup.py`、`pyproject.toml`、`requirements*.txt`、`format.sh`、`mypy.ini`、`codecov.yml`、`CMakeLists.txt`、`cmake/`、`DCO`、`LICENSE`、`CODE_OF_CONDUCT.md`、`CONTRIBUTING.md`、`README.md`、`README.zh.md`、`collect_env.py`、`packages.txt`、`.pre-commit-config.yaml`、`.readthedocs.yaml`、`typos.toml`、`.gemini/`）。这些与 Neutron (Rust) 无关，应在新分支上 `git rm` 清理，保留 `LICENSE`（Apache-2.0 通用）。

## Continuity Log（进度遗言）

每轮工作结束在此追加一段，记录：当前状态 / 已完成 / 下一步。供下一任 agent 接力。

### 2026-07-09 — P0 后恢复 + 回归验证（feat/recover-and-verify）

**当前状态**：9-crate workspace 全部建好并编译通过，回归全绿。

**已完成**：
- base (raw.rs packed buffer + lib.rs Safe Graph) — commit 434f3d0
- lisp (parser + interp) — commit 32fa7ad
- common (Target/OptLevel/IdGen/Arena/Config/dump_graph) — commit b1d6ae9
- frontend (onnx/dsl/pt 占位) — commit 03536f1
- optimizer (三阶段 pipeline + passes/algebra/float_opts/cse/decompose/fuse/cost_model) — commit fabaf6d
- arch (ArchGraph + lower + cuda/npu 描述) — commit c1c1511
- isel (select 占位) — commit c1c1511
- interface (compile API + 端到端测试) — commit c1c1511
- cli (neutron 二进制) — commit c1c1511
- 修复 3 处 unused import + common::Target Default + lisp::Val PartialEq — commit c1c1511
- cargo fmt 应用 — commit 581e6f3

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：11 passed (common 2 + interface 1 + lisp 4 + optimizer 4)
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常，pipeline 串联成功

**下一步**（优先级排序）：
1. 合并 feat/recover-and-verify → dev → main
2. 新分支清理 vllm-ascend 残留文件（见上"待办"）
3. 充实优化 pass 实现（当前 algebra 只做了 x+0/x*1，float_opts/decompose/fuse 多为 TODO 占位）：
   - algebra：补 x*1、常量折叠
   - float_opts：实现 FastInvSqrt / SoftmaxOnline / MulByTwoToAdd / DivByConstToMul
   - decompose：实现 LayerNorm / Softmax / Gelu 一对多拆细
   - fuse：实现多对一启发式融合（带 cost model 判定收益）
4. frontend：实现真正的 ONNX 解析（当前空输入返回 Placeholder）
5. isel：用 lisp 解释器驱动 isel 规则匹配
