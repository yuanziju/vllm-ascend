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

> 本仓库 git 历史从 vllm-ascend fork 而来。**已于 2026-07-09（feat/cleanup-and-basic-passes 分支）清理完 vllm-ascend 残留 460 文件**（`vllm_ascend/`、`csrc/`、`docs/`、`tests/`、`benchmarks/`、`examples/`、`tools/`、`.github/`、`Dockerfile*`、`setup.py`、`pyproject.toml`、`requirements*.txt`、`format.sh`、`mypy.ini`、`codecov.yml`、`CMakeLists.txt`、`cmake/`、`DCO`、`CODE_OF_CONDUCT.md`、`CONTRIBUTING.md`、`README.md`、`README.zh.md`、`collect_env.py`、`packages.txt`、`.pre-commit-config.yaml`、`.readthedocs.yaml`、`typos.toml`、`.gemini/`），仅保留 `LICENSE`（Apache-2.0）。此条已结案。

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

### 2026-07-09 — 残留清理 + 基础 pass 充实（feat/cleanup-and-basic-passes）

**当前状态**：vllm-ascend 残留已清空，4 个基础优化 pass（algebra/cse/float_opts/fuse）从空壳补全为可用实现，回归全绿。本分支待合并回 dev → main。

**已完成**：
- 清理 vllm-ascend 残留 460 文件，仅留 LICENSE，重写 README.md（Neutron 版）— commit 9c42f28
- algebra 补全：常量折叠 + x+0/x*1/x*0=0/x/1=x + 可选 x-x=0/x/x=1（unsafe 开关）。两阶段模式（先收集建议再应用）解决 borrow 冲突；不动点迭代 + processed HashSet 防死循环。6 单测 — commit a387a38
- CSE 升级指纹：`Fingerprint` enum（Op(op, normalized_inputs) / Constant(f64::to_bits)），可交换 op (Add/Mul) inputs 排序后比较。4 单测 — commit 7d6d364
- float_opts 实现：DivByConstToMul（x/c→x*(1/c) 改 op_tag+换常量）、MulByTwoToAdd（x*2→x+x）、FastInvSqrt/SoftmaxOnline 仅识别不改图（留给 lowering）。base 加 Sqrt(20)/Exp(21)/Pow(22) op。5 单测 — commit fb2f81d
- fuse 实现：elementwise 链（Add/Sub/Mul/Div/Relu/Gelu/Sigmoid/Tanh/Sqrt/Exp）尾节点改 Custom，inputs 重写为链头 inputs，attr 记 op 序列，其余节点 compact 删除。cost_model 判定融合收益 > 0 才融。3 单测

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：27 passed (common 2 + interface 1 + lisp 4 + optimizer 20)

**设计哲学遵守**：本轮严格按"不做固定模式匹配，先把基础优化做扎实"的约束。algebra 只用简单代数规则；CSE 是 IO 同样性识别；float_opts 针对浮点本身结构；fuse 是 elementwise 链通用融合（非 MatMul+Add→Linear 这类贪心模式）。decompose（LayerNorm/Softmax/Gelu 一对多拆细）刻意推迟到后续轮，先把基础打牢。

**下一步**（优先级排序）：
1. 合并 feat/cleanup-and-basic-passes → dev → main（按"没崩就合并"策略）
2. decompose 从经典到难：LayerNorm → Softmax → Gelu（每轮一个，单测驱动）
3. frontend：实现真正的 ONNX 解析，构造带算子的测试 ONNX
4. isel：用 lisp 解释器驱动 isel 规则匹配
5. algebra 可继续扩展：常量传播、shape 推断后基于 shape 的简化

### 2026-07-09 — decompose 三件套实现（feat/neutron-c3d30820）

**当前状态**：decompose 三个复合算子（LayerNorm/Softmax/Gelu）全部从空壳补全为数学等价的细粒度拆分，回归全绿。本分支待合并回 main。

**分支策略调整**：按用户要求改为"公共前缀+哈希"命名（`feat/neutron-<8hex>`），能复用就复用，不建一堆分支。

**已完成**：
- base 补 IR op：ReduceSum(23)/ReduceMean(24)/ReduceMax(25) + from_u8 映射。decompose 拆细需要 Reduce 类原语
- decompose LayerNorm：LN(x,γ,β,ε) → mean=ReduceMean(x) → xc=Sub(x,mean) → var=ReduceMean(xc*xc) → std=Sqrt(var+ε) → inv=Div(1,std) → norm=Mul(xc,inv) → scaled=Mul(norm,γ) → out=Add(scaled,β)。10 个原语节点
- decompose Softmax（数值稳定版）：m=ReduceMax(x) → shifted=Sub(x,m) → e=Exp(shifted) → s=ReduceSum(e) → out=Div(e,s)。5 个原语节点。用 max 技巧避免 Exp 溢出
- decompose Gelu（tanh 近似）：x³=x*x*x → kx3=Mul(x3,0.044715) → t=Add(x,kx3) → ct=Mul(t,sqrt(2/π)) → th=Tanh(ct) → 1+th → 0.5*x → out=Mul(0.5x, 1+th)。9 个原语节点
- 通用辅助：build_reduce/build_binop/build_unop 构造子图；read_axis/read_epsilon 读属性；rewrite_value_uses 重写原节点输出的所有使用者；compact 物理删除原节点
- cost_model 补 ReduceSum/ReduceMean/ReduceMax 估算（flops = in_bytes/4*2）
- lowering 补全原语 op 映射（Sub/Mul/Div/Sqrt/Exp/Pow/Reduce*/Constant 等），确保 decompose 后图能 lower
- 5 单测：layernorm_decomposes_to_subgraph / softmax_decomposes_numerically_stable / gelu_decomposes_to_tanh_approx / output_rewired_to_new_subgraph / non_decomposable_nodes_untouched

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：32 passed (common 2 + interface 1 + lisp 4 + optimizer 25)
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：decompose 是"一对多拆细"（合法），把复合算子拆成基础原语让后续 algebra/CSE/fuse 在细粒度上做通用优化，不是贪心模式匹配。Gelu 用 tanh 近似避免 erf（erf 后端不一定有原生 kernel）。Softmax 用数值稳定 max 技巧。

**下一步**（优先级排序）：
1. 合并 feat/neutron-c3d30820 → main
2. frontend：实现真正的 ONNX 解析，构造带算子的测试 ONNX（让上层有真实输入喂给 decompose + 优化器）
3. isel：用 lisp 解释器驱动 isel 规则匹配
4. algebra 扩展：常量传播、shape 推断后基于 shape 的简化
5. fuse 可扩展：reduce + elementwise 融合（目前只做 elementwise 链）
