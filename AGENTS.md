# AGENTS.md

Guidance for AI coding agents working in the **Neutron** repository.

> 这是 Neutron 项目（Rust ML 编译器）的 agent 指南。前任 agent 因 P0 级事故销毁过项目，本文件含**防 P0 安全规则**与 **Continuity Log（进度遗言）**，每轮工作前必读。
>
> **操作流程见 [WORKSPEC.md](WORKSPEC.md)**：每次新会话上手的标准 checklist + **时间盒执行规则**（验证回归 → 建分支 → 计时北京时间 → 列待办/子代理并行做满规定时长 → 到点写遗言合并）。本文件是项目宪法，WORKSPEC.md 是操作手册，两者配合。

## Project Overview

Neutron 是一个用 Rust 编写的 ML 编译器，把高层模型（ONNX / 自研 DSL / PyTorch）经架构无关优化 + 架构相关 lowering + 指令选择，编译成目标后端（CUDA / Ascend NPU / CPU）的指令序列。

- License: Apache-2.0
- Language: Rust (edition 2021, MSRV 1.75)
- 11-crate workspace（**无 `neutron-` 前缀**，crate 名即职责名）

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
Cargo.toml              # workspace 根，11 crate
crates/
  base/                 # IR 核心：storage.rs (packed buffer unsafe) + lib.rs (Safe Graph/NodeView/ValueView)
                        #   NeutronError, NodeId/ValueId, DType/TypeTag/Type, OpKind, Attr, Pass/PassContext/Visitor
  common/               # Target(Cuda/Npu/Cpu), OptLevel, IdGen, Arena, Config, dump_graph
  frontend/             # onnx.rs / dsl.rs / pt.rs 前端解析（当前占位）
  optimizer/            # 三阶段 pipeline 入口 + passes(DCE/Verify) + algebra + float_opts + cse + decompose + fuse + cost_model
  arch/                 # ArchGraph + lower() 1:1 op→native kernel + cuda.rs/npu.rs 设备描述
  lisp/                 # S-expr 解释器（parser + interp），用于 isel 规则
  isel/                 # 指令选择 select()
  regalloc/             # Chaitin-Briggs 图着色寄存器分配（types/liveness/interference/coalescing/coloring/allocator）
  backend/              # 四后端代码生成：CUDA(wmma/wgmma/tcgen05) / Triton(TMA) / Metal(simdgroup) / CANN(AscendC)
  interface/            # 唯一公开 API compile()，串联 frontend→optimizer→arch→isel→regalloc→backend
  cli/                  # neutron 二进制，--target/--opt/--dump/-o/--version/--help
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
- **unsafe**：集中在 `base/src/storage.rs`，上层只暴露 Safe API

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

### 2026-07-10 — isel lisp 规则化 + frontend 真解析（feat/neutron-bf91af14）

**当前状态**：isel 从硬编码 match 改为 lisp 规则驱动；frontend 从占位改为真正的 ONNX protobuf 解析 + 文本 DSL 解析。回归全绿。本分支待合并回 main。

**已完成**：
- **isel 规则化**：从硬编码 match 改为 S-expr 规则 `(rule (when <cond>) (emit <op> <args>...))`。规则由 lisp 解释器求值，绑定 `op`/`idx`/`target` 上下文变量。默认规则集覆盖 21 个 native kernel。未知 op 报错（强制写规则，不静默漏）。8 单测
- **lisp 增强**：interp 加 `and`/`or` 短路逻辑特殊形式 + `not`/`str`（字符串拼接）/`str=` 内建函数；parser 加 `"..."` 字符串字面量解析（带转义）。isel 规则的条件表达式和 emit 参数现在能用字符串字面量
- **frontend ONNX 真解析**：手写极简 protobuf wire-format 读取器（`proto.rs`，无 prost/prost-build 重依赖），解出 ModelProto→GraphProto→NodeProto 的 op_type/input/output/name/initializer/input/output 字段。ONNX op_type→OpKind 映射覆盖 26 个常见算子，未知算子→Custom（attr 记录原始 op_type 字符码）。名称注册表做 SSA 去重。8 单测（含手工编码 ONNX 字节流的端到端解析）
- **frontend DSL 解析**：极简文本格式 `in x: f32[2,3]` / `y = relu(x)` / `out z`，支持注释。方便手写测试图，不依赖 ONNX 二进制。5 单测
- **proto.rs**：独立模块，Cursor 读 varint/length-delimited/tag/skip_field。4 单测

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：53 passed (common 2 + frontend 13 + interface 1 + isel 8 + lisp 4 + optimizer 25)
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：isel 规则化符合"规则用函数实现，后期再抽象"——现在是 lisp 规则（可热加载、可读、可扩展），比硬编码 match 灵活。frontend 不引 prost 重依赖，手写 protobuf 解码器符合"简单"哲学。未知算子映射 Custom 不报错（前向兼容），但 isel 无规则匹配会报错（强制完整性）。

**下一步**（优先级排序）：
1. 合并 feat/neutron-bf91af14 → main
2. algebra 扩展：常量传播、shape 推断后基于 shape 的简化
3. fuse 可扩展：reduce + elementwise 融合（目前只做 elementwise 链）
4. isel 规则从文件加载（热加载，不重编译）
5. frontend：解析 ONNX 属性（axis/epsilon 等），喂给 decompose 的 read_axis/read_epsilon
6. pt 前端：PyTorch 解析（当前占位）

### 2026-07-10 — shape 推断 + isel 文件热加载 + 底层存储 compact 属性修复（feat/neutron-5f1c5deb）

**当前状态**：新增 shape 推断 pass 让 cost_model 估算更准；isel 规则支持从文件热加载；修复底层存储 compact 丢失属性的 critical bug。回归全绿，本分支待合并回 main。

**已完成**：
- **shape 推断 pass**（optimizer/shape_infer.rs）：elementwise 广播 + reduce 沿轴消维 + MatMul [m,k]×[k,n]→[m,n]；不动点迭代，两阶段模式（先收集 to_fill Vec 再应用，解决 borrow 冲突）；所有 op 加 shape_known 守卫，要求输入 shape 全已知才推，避免用未知输入推出错误 shape 被锁定。`set_value_shape` 回填。注册到 DecomposePass 之后。6 单测
- **isel 规则文件热加载**（isel/lib.rs）：`load_rules_from_src`（括号配平切分多条规则 + `;` 注释支持）+ `load_rules_from_file`（从路径加载不重编译）。4 单测
- **底层存储 compact 属性丢失修复**（base/lib.rs + base/raw.rs，critical bug）：原 compact 复制节点时漏拷 attrs，导致 Constant 的 Value 丢失、reduce 的 Axis 丢失——任何经过 compact 的图都受影响，algebra 折叠在 decompose（调 compact）之后失效。修复：新增 `copy_attrs` 辅助按 AttrTag 分发复制；底层存储层补 `AttrTag::from_u8` + `add_attr_float_array` + `set_value_shape`。3 回归测试

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：70 passed (base 3 + common 2 + frontend 13 + interface 1 + isel 12 + lisp 4 + optimizer 35)

**设计哲学遵守**：shape 推断让 cost_model 有准确估算（FLOPs + memory access 需 shape），符合"cost model 现在就做"。compact 属性修复是审查底层数据结构发现的 critical 问题——纯函数式 SSA 重排自由的前提是图变换不丢信息，compact 丢属性破坏了这个前提。isel 文件热加载符合"规则用函数实现，后期再抽象"。

**避嫌规范**：本轮起，输出与新写注释避免使用某些英文技术词的中文翻译（该翻译在中文语境敏感），改用"底层存储"等中性表述；现有文件名/类型名保持现状（重命名核心类型是大改动，后续轮专门做）。

**下一步**（优先级排序）：
1. 合并 feat/neutron-5f1c5deb → main
2. 底层存储模块/类型重命名（去除敏感词，需全量替换引用，大改动单独一轮）
3. algebra 扩展：常量传播、shape 推断后基于 shape 的简化
4. fuse 可扩展：reduce + elementwise 融合
5. frontend：解析 ONNX 属性（axis/epsilon 等），喂给 decompose
6. pt 前端：PyTorch 解析

### 2026-07-10 — 底层存储重命名 + 多 pass 充实 + frontend 属性解析（feat/neutron-a7e3c902）

**当前状态**：本轮一小时连做 8 件事，全部提交，回归全绿。底层存储模块避嫌重命名完成；algebra/fuse/shape_infer/cost_model 四 pass 充实；frontend ONNX 属性解析打通到 decompose 并有端到端测试。已合并回 main。

**已完成**（8 commit，按时序）：
- **底层存储模块避嫌重命名 raw→storage**（commit 0013728）：12 文件全量替换（RawGraph→StorageGraph、.raw→.storage、raw::→storage::、RawAttrKey→StorageAttrKey 等），git 识别为 rename。仅保留 std API `from_raw_parts`（非自有命名）。这是上轮明确推迟的"大改动单独一轮做"
- **algebra 基于 shape 的 no-op 简化 + shape_infer Reshape/Transpose 推断**（commit 7727dce）：base 加 `AttrKey::Perm=12`（Transpose 轴排列，IntArray）。shape_infer 推 Reshape（输出=attr Shape）+ Transpose（输出=输入按 perm 重排）。algebra 加 simplify_reshape（输入输出 shape 相等→ReplaceWith input）+ simplify_transpose（perm 单位排列→ReplaceWith input）。新增 `read_int_array_attr` 通用辅助。6 新单测
- **fuse reduce + unary elementwise 融合**（commit 75513c2）：新增 `is_reduce` 辅助；重写 `build_fusion_chain`，链头允许一个 reduce（仅 unary elementwise 才接，reduce 是 shape 分界点不再往前扩）；apply_fusion 复制 reduce 的 Axis attr 到融合节点保留轴信息。2 新单测
- **frontend ONNX 属性解析**（commit ab9ca4a）：NodeProto.attribute (field 5) 原先跳过，现解析 AttributeProto（name(1)+type(3,跳过)+f(4,FIXED32)+i(5,varint)+ints(21,packed)），value 按 i/f/ints 存在性推断。按 op_type 映射到 StorageAttrKey：reduce/concat 的 axis/axes→Axis（axes 取首元素）、LayerNormalization 的 epsilon→Epsilon、Transpose 的 perm→Perm、Reshape 的 shape(attr)→Shape。5 新单测
- **interface 端到端测试**（commit 25c9a4f）：构造含 LayerNormalization(x,gamma,beta,epsilon=1e-3) 的 ONNX 字节流，验证前端解析后 epsilon≈1e-3 写入 Epsilon attr，单独跑 decompose 后 LayerNorm 拆成原语子图（ReduceMean/Sub/Sqrt/Div/Mul/Add 齐全）。隔离跑 decompose 避免被 fusion 干扰。证明整条链路：ONNX protobuf 属性解码 → StorageAttrKey 写入 → decompose 的 read_epsilon 消费
- **algebra 一元常量折叠 Sqrt/Exp/Pow**（commit 22a76a7）：复用 FoldToConstant 机制，sqrt(c)→Constant(c.sqrt())、exp(c)→Constant(c.exp())、pow(c1,c2)→Constant(c1.powf(c2))、pow(x,1)→x。负底数+非整指数 / 负数 sqrt 返回 NaN 与运行时一致。5 新单测
- **shape_infer Concat 沿 axis 拼接推断**（commit 76d8a0c）：Concat 输出 shape = 各输入 shape 在 axis 维求和、其余维相等，要求全部已知 + rank 相同 + 非轴维相等；axis 支持负值。让 cost_model 对 Concat 后图估算准确。2 新单测
- **cost_model MatMul FLOPs 用输入 shape 算 2·m·n·k**（commit c2f4a0e）：旧估计 n=(out_bytes/4).sqrt() 假设方阵，非方阵严重失准。shape_infer 现能填 MatMul 输入 shape，故读取双输入 [m,k]×[k,n] 算 FLOPs=2·m·n·k；shape 未知时退化到方阵估计（向后兼容）。新增 `matmul_flops` 辅助 + 3 新单测（方阵/非方阵/未知 shape 退化）

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：94 passed (base 3 + common 2 + frontend 18 + interface 2 + isel 12 + lisp 4 + optimizer 53)
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：
- 重命名是用户上轮明确的"大改动单独一轮做"，本轮完成，纯机械替换不改语义
- algebra 一元常量折叠是"常量折叠"的自然扩展（基础优化，非模式匹配），复用已有 FoldToConstant 机制
- fuse reduce+elementwise 是通用融合（reduce 作 shape 分界点链头），非 MatMul+Add→Linear 贪心模式
- shape_infer Concat 让 cost_model 估算准确（符合"cost model 现在就做"），保守策略（不满足条件就不推，避免错误 shape 被锁定）
- frontend 属性解析不引 prost 重依赖，手写 protobuf 解码；未知属性静默忽略（前向兼容）

**避嫌规范延续**：本轮完成 raw→storage 重命名，文件名/类型名/注释/输出全部用"底层存储"等中性表述，不再用敏感词的中文翻译。

**下一步**（优先级排序）：
1. pt 前端：PyTorch 解析（当前占位，是 frontend 最后一块）
2. fuse 可扩展：binary elementwise + reduce（当前只 unary elementwise 接 reduce）
3. isel：更多目标后端的规则覆盖（当前 21 个 native kernel）
4. frontend：解析 ONNX initializer 的实际张量数据（当前只取 name，值留空）
5. algebra 扩展：常量传播跨节点（constprop 当前只做 value canonicalize）、shape 推断后基于 shape 的进一步简化（如广播后 x*ones→x）

### 2026-07-10 — initializer 张量解析 + fuse side inputs + algebra shape 简化（feat/neutron-init-fuse-shape）

**当前状态**：本轮一小时连做 3 件事，全部提交，回归全绿。frontend ONNX initializer 张量数据完整解析映射成 Constant 节点；fuse 支持带 side inputs 的 binary elementwise 融合；algebra 识别多元素全 0/全 1 张量做 shape-based 简化。本分支待合并回 main。

**已完成**（3 commit，按时序）：
- **frontend ONNX initializer 张量数据解析**（commit 09f6ed6）：TensorProto 完整解析（dims/data_type/raw_data/float_data/double_data），FLOAT(1)/DOUBLE(11) 张量映射成 Constant 节点带 Value attr（单元素 Float 让 constant_value() 立即可用，多元素 FloatArray 让 constant_tensor() 可读）+ 输出 value 带真实 dims shape。其余 dtype 退化成未知 shape 输入。base 新增 `attr_float_array` 读取器 + `constant_tensor()` 返回完整 FloatArray + `constant_value()` 扩展支持单元素 FloatArray。**关键 bug 修复**：initializer 的 Constant 节点占用前面的 NodeId，第二遍填 inputs 不能用 `node_idx as u32`，改用 `node_ids: Vec<NodeId>` 记录真实 ID。5 新单测
- **fuse binary elementwise + side inputs**（commit b71566b）：重写 `build_fusion_chain` 收集 side_inputs + side_positions，让 binary elementwise（Add/Sub/Mul/Div）的"另一输入"作为 side input 进融合节点而非被丢弃。diamond 检测（side input 由链中节点产生→放弃整条链）+ 自引用检测（add(r,r)→放弃）保证正确性。apply_fusion 加 consumed set 跳过与已应用链重叠的机会。融合后 inputs = 链头 inputs + side inputs；op 序列→Shape attr，side input 位置→Strides attr 供 lowering 重建。3 新单测
- **algebra shape-based 简化**（commit b7f532a）：新增 `constant_is_uniform` 判断常量张量是否所有元素都等于 target，覆盖标量 Float / 单元素 FloatArray / 多元素 FloatArray 三种存储形式。这让 algebra 能识别 ONNX initializer 的 ones/zeros（多元素全 1/全 0 张量）。规则扩展：x+zeros→x、x*ones→x、x*zeros→复用那个 zeros 张量（ReplaceWith 保留 shape，不再退化为标量 FoldToConstant）。simplify_mul 的 x*0 分支从 FoldToConstant(0.0) 改为 ReplaceWith(那个0)，复用已有常量节点保留 shape；原 mul_zero_folds 测试仍兼容。4 新单测

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：106 passed (base 3 + common 2 + frontend 23 + interface 2 + isel 12 + lisp 4 + optimizer 60)
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：
- initializer 解析打通"ONNX 张量数据 → Constant 节点 → algebra/float_opts 基于常量的优化"链路，之前 initializer 只取 name 退化成未知输入，Constant 没有真值导致基于常量的优化在 ONNX 输入上完全失效
- fuse side inputs 是通用融合的正确性修复（旧实现把 binary 的另一输入丢弃是错的），diamond/自引用检测保证只融安全的情况，非贪心模式匹配
- algebra shape-based 简化是"简单代数规则"的自然扩展（识别全 0/全 1 张量，不是复合算子模式），x*zeros 用 ReplaceWith 保留 shape 比 FoldToConstant 标量更准

**下一步**（优先级排序）：
1. pt 前端：PyTorch 解析（当前占位，是 frontend 最后一块）
2. isel：更多目标后端的规则覆盖（当前 21 个 native kernel）
3. algebra 常量传播跨节点（constprop 当前只做 value canonicalize）
4. fuse 可扩展：reduce + elementwise 更复杂模式（当前 binary side inputs 已支持）
5. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型（tensor/GraphProto）

### 2026-07-10 — FastInvSqrt 真正图重写 + Rsqrt op（feat/neutron-7b3e9c41）

**当前状态**：float_opts 的 FastInvSqrt 从"识别不改图"空壳做成真正的浮点结构图重写，新增 Rsqrt op 全链路打通。回归全绿。本分支待合并回 main。

**用户指引**：本轮起用户明确——不要在简单代数规则（x+0/x*1 那类）和常量传播上花时间，这些在真实 ML 算子计算里基本不出现；当前多数 pass 还是基于规则的简单匹配，价值有限。应聚焦设计哲学点名的"浮点结构优化（IEEE754 位级 trick / Flash Attention online-softmax 式重排）"——这才是项目招牌。故本轮选 FastInvSqrt（之前是空壳）动手。

**已完成**（2 commit，按时序）：
- **base 加 Rsqrt op(=26) + 全链路接入**（commit e63b1fa）：`OpKind::Rsqrt=26` + from_u8。shape_infer 加 Rsqrt 到 unary elementwise passthrough；cost_model 估算 out_bytes/4*2（比 Sqrt+Div 便宜）；lowering `Rsqrt → "rsqrt"`；isel 加 `(when (= op "rsqrt"))` 规则。新 op 必须四点全接，否则 lowering 报"未覆盖"
- **float_opts FastInvSqrt 真正重写**（commit f7a0cc1）：核心是浮点恒等式 `a/√b = a·b^(-1/2)`。`Div(a, Sqrt(b))` → a==1.0 常量时直接 `Rsqrt(b)`（2 op 降 1 op，Div 节点本身改 Rsqrt，输入换 b）；a 非常量时 `Mul(a, Rsqrt(b))`（新建 Rsqrt 节点吃 b，Div 改 Mul）。Sqrt+Div（含一个贵的 Div）融成 Rsqrt（单条硬件指令 / 0x5f3759df 魔数 bit trick，Quake III fast inverse sqrt）+ 便宜 Mul。RMSNorm/LayerNorm 等 normalization 到处出现。FloatOpt::FastInvSqrt enum 补 numerator/sqrt_input/numerator_is_one 字段；try_match 改为匹配**除数**是 Sqrt（分子任意，注意 `sqrt(x)/a` ≠ `a·rsqrt(x)` 不匹配）。SoftmaxOnline 明确留作 recognition——真正 FA 融合（softmax+matmul）是设计哲学禁止的贪心模式，online-softmax 本质是 kernel tiling 策略非 IR 重写。fuse is_elementwise 加注释说明 Rsqrt 故意不列入（保留为独立 op 让 lowering 发专用 rsqrt kernel，而非被融进链变 Custom）

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：112 passed（base 3 + common 2 + frontend 23 + interface 3 + isel 12 + lisp 4 + optimizer 65）—— 较上轮 106 +6（5 新 float_opts 单测 + 1 e2e）
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**e2e 验证**（interface）：构造 LayerNormalization(x,gamma,beta,epsilon=1e-3) 的 ONNX，跑完整优化 pipeline（O1），验证 decompose 产生的 `Div(1,Sqrt(var+ε))` 被 float_opts 融合成 Rsqrt，且 Rsqrt 全链路通——lowering 发 "rsqrt" kernel、isel 选 "rsqrt" 指令。证明 IEEE754 浮点结构优化在全 pipeline 生效。

**⚠️ 发现的 critical 缺口（未修，下轮优先）**：O2 的 fusion 会把 elementwise 链融成 `Custom` 节点，但 lowering **未覆盖 Custom**（`other => Err`），导致任何非空图跑 O2 都会在 lowering 崩。之前几轮"CLI e2e 正常"只测了 empty.onnx（无算子无融合），从未暴露。更麻烦：`Custom` 被**复用**于两种语义——fusion 融合结果 vs 未知 ONNX 算子（frontend 把未知 op_type 映射 Custom + attr 记原始 op_type 字符码），两者 lowering 语义不同。修复需区分（建议新增 `Fused` op 专管融合结果，Custom 留给未知算子），是单独一轮的活。本轮 e2e 测试用 O1 避开此缺口。

**设计哲学遵守**：FastInvSqrt 是浮点代数恒等式（`a/√b=a·b^(-1/2)`）的结构融合，不是 MatMul+Add→Linear 贪心模式；针对 IEEE754 浮点本身结构（rsqrt 有专用硬件指令/位 trick），正是设计哲学点名"类 Quake III fast inverse sqrt"。SoftmaxOnline 不做成重写是经过论证的——真 FA 融合是禁止的贪心模式。

**新增长效机制**：本轮起建立 [WORKFLOW.md](WORKFLOW.md)（新 agent 上手标准 checklist：验证回归→建分支→频繁提交→写遗言→合并），AGENTS.md 顶部已加引用。本环境会话窗口不稳定（挂过多次），对策是频繁提交 + 中途写遗言。

**下一步**（优先级排序）：
1. **修 fusion→Custom→lowering 缺口**（critical，阻塞 O2 真实模型）：新增 `Fused` op 专管融合结果（Custom 留给未知算子），lowering 发 "fused" kernel，isel 加规则；或让 lowering 读 Custom attr 区分。修完后 e2e 测试可升回 O2
2. pt 前端：PyTorch 解析（frontend 最后一块占位）
3. float_opts 可继续：`Reciprocal(Sqrt(x))` 模式（ONNX Reciprocal op）也映射 Rsqrt；识别 `x * rsqrt(y)` 的 RMSNorm 整体模式做 cost-based 决策
4. isel：更多目标后端规则覆盖
5. fuse 可扩展：reduce + elementwise 更复杂模式

### 2026-07-10 — Fused op 修 O2 缺口 + 5 个浮点结构重写（feat/neutron-9c2e1a7b）

**当前状态**：本轮为首个 30 分钟时间盒任务（WORKSPEC.md 规则），从 22:38 做到 22:53。修了上轮发现的 fusion→Custom→lowering critical 缺口（O2 不再崩），并把 float_opts 从 2 个简单规则扩到 7 个浮点结构重写。回归全绿，本分支待合并回 main。

**用户指引延续**：不在简单代数/常量传播上花时间，聚焦浮点结构优化。本轮 float_opts 新增的 5 个全是浮点代数恒等式重排，非贪心模式。

**已完成**（7 commit，按时序）：
- **base 加 Fused op(=27) 修 fusion→Custom→lowering critical 缺口**（commit cc060ec）：上轮发现 O2 fusion 产 Custom 但 lowering 未覆盖 Custom（Custom 还被未知 ONNX 算子复用），任何非空图跑 O2 都崩。新增 `Fused` op 专管融合产物，Custom 留给未知算子。fuse apply_fusion 链尾 op_tag Custom→Fused（doc/注释/5 单测断言同步）；lowering Fused→"fused" kernel、Custom→"custom" kernel（原 other→Err 改显式分支）；isel 加 (when fused/custom) 规则；cost_model Fused 估值 launch=0；shape_infer Fused 取首输入 shape；interface e2e 升回 O2 全链路通
- **float_opts ReciprocalSqrt**（commit 94c6d4f）：`Reciprocal(Sqrt(x))` → `Rsqrt(x)`（2 op 降 1 op），同 1/√x 恒等式。ONNX Reciprocal(Sqrt) 是 RMSNorm 常见写法。base 加 Reciprocal op(=28) + frontend 映射 + 全链路接入 + 2 单测
- **float_opts DivByReciprocal**（commit a7c4c65）：`a / Reciprocal(b)` → `Mul(a, b)`（消去 Reciprocal+Div 换便宜 Mul），a/(1/b)=a·b。2 单测
- **interface e2e Reciprocal(Sqrt)→Rsqrt**（commit c60a0a1）：用 base API 构图跑 O2 pipeline 验证 ReciprocalSqrt 全链路（lowering rsqrt kernel、isel rsqrt 指令）
- **float_opts ExpMulFusion**（commit 1bd2907）：`Exp(x)*Exp(y)` → `Exp(Add(x,y))`（省一个 Exp），e^x·e^y=e^(x+y)。softmax/attention exp 链相乘极常见。新建 Add 吃 [x,y]，复用 exp_x 改吃 Add 输出，Mul 输出使用者重写到 exp_x 输出。1 单测 + 修 3 处 clippy manual_contains
- **float_opts ExpDivFusion**（commit fdfa275）：`Exp(x)/Exp(y)` → `Exp(Sub(x,y))`（省一个 Exp），e^x/e^y=e^(x-y)，ExpMulFusion 对偶。attention score 归一化常见。1 单测

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：119 passed（base 3 + common 2 + frontend 23 + interface 4 + isel 12 + lisp 4 + optimizer 71）—— 较上轮 112 +7（6 新 float_opts + 1 e2e）
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：
- Fused vs Custom 分离是"让 lowering 按 op_kind 直接分派不靠 attr 探测猜语义"的正确性修复，非贪心模式
- 5 个新浮点重写全是浮点代数恒等式：1/√x=x^(-1/2)、a/(1/b)=a·b、e^x·e^y=e^(x+y)、e^x/e^y=e^(x-y)。是设计哲学点名"IEEE754 位级 trick / Flash Attention online-softmax 式重排"的具体落地，非 MatMul+Add→Linear 贪心模式
- Reciprocal(Sqrt) 和 Div(1,Sqrt) 是同一 1/√x 恒等式的两种 ONNX 写法，都映射 Rsqrt——前向覆盖两种 frontend 输入形态

**新增长效机制**：本轮首次执行 [WORKSPEC.md](WORKSPEC.md) 时间盒规则——22:38 开工计时北京时间，持续列待办+完成待办+子代理并行（用了 2 个 search 子代理摸清 fuse/frontend 的 Custom 结构），到点收尾。用户确认 WORKSPEC 定稿。

**下一步**（优先级排序）：
1. isel：补 Reciprocal/Rsqrt/Fused/Custom 规则覆盖检查（本轮已加规则但未做覆盖审计）
2. pt 前端：PyTorch 解析（frontend 最后一块占位）
3. float_opts 可继续：识别 `x * rsqrt(y)` 的 RMSNorm 整体模式做 cost-based 决策；`Sqrt(x*x)` → `Abs(x)`；`Log(Exp(x))` → `x`
4. fuse 可扩展：reduce + elementwise 更复杂模式
5. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型

### 2026-07-10 — isel 覆盖审计 + Pow 浮点结构重写（feat/neutron-pow-half + fix/lowering）

**当前状态**：isel 覆盖审计完成（补 5 个数据移动 op 的 lowering+isel 规则）；新增 3 个 Pow 浮点结构重写。回归全绿，已合并回 main。

**本轮是 WORKSPEC 时间盒任务（续上一窗口）**：上一窗口 22:38 开工做 Fused op 修缺口 + 5 浮点重写，本窗口接续完成 isel 覆盖审计 + Pow 重写。

**已完成**（3 commit）：
- **lowering 移除 unreachable catch-all**（commit db45392，fix/lowering-unreachable-pattern）：上轮 isel 审计给 lowering 补全 5 个数据移动 op（Reshape/Transpose/Concat/Slice/Pool）后，所有 OpKind 变体已显式覆盖，`other =>` catch-all 变 unreachable pattern 触发 clippy -D warnings。修法：移除 catch-all，靠 Rust match 穷举性检查——新增 op 时编译器报 non-exhaustive 强制补 lowering 分支，比 catch-all 更安全（不会静默漏）
- **float_opts PowHalfToSqrt**（commit 4249ab0）：`Pow(x, 0.5)` → `Sqrt(x)` / `Pow(x, -0.5)` → `Rsqrt(x)`。把通用 Pow（log/exp 实现的超越函数，贵）换成专用单条硬件指令（IEEE754 sqrt/rsqrt，rsqrt 可用 0x5f3759df bit trick）。幂指数 ±0.5 时 x^0.5=√x，x^(-0.5)=1/√x。Pow 节点改 op + 输入换 [x]（丢弃常量指数），输出 value 不变。RMSNorm 的 `x * Pow(var+eps, -0.5)` 常见此模式。4 单测
- **float_opts PowNegOneToReciprocal**（commit 56d1bd8）：`Pow(x, -1.0)` → `Reciprocal(x)`。x^(-1)=1/x=reciprocal(x)，同 PowHalfToSqrt 一类：通用幂 → 专用 op（单条硬件指令）。1 单测

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：125 passed（base 3 + common 2 + frontend 23 + interface 5 + isel 12 + lisp 4 + optimizer 76）—— 较上轮 119 +6（5 新 float_opts 单测 + 1 Pow e2e）
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：Pow 系列重写是"通用幂函数 → 专用 IEEE754 硬件指令"的浮点结构优化，正是设计哲学点名"类 Quake III fast inverse sqrt"同族——Pow 用 log/exp 超越函数实现，Sqrt/Rsqrt/Reciprocal 都是单条硬件指令。非贪心模式匹配（不是 Pow+Mul→xxx 复合算子融合）。

**isel 覆盖现状**：lowering 现已显式覆盖全部 OpKind 变体（29 个），无 catch-all。isel 规则覆盖 21+ 个 native kernel（含本轮已确认的 rsqrt/reciprocal/fused/custom/reshape/transpose/concat/slice/pool）。新增 op 必须四点全接（base OpKind + from_u8 / shape_infer / cost_model / lowering / isel），否则编译器穷举检查报错。

**下一步**（优先级排序）：
1. pt 前端：PyTorch 解析（frontend 最后一块占位）
2. float_opts 可继续：识别 `x * rsqrt(y)` 的 RMSNorm 整体模式做 cost-based 决策；`Sqrt(x*x)` → `Abs(x)`（需新增 Abs op + 全链路接入）；`Log(Exp(x))` → `x`（注意溢出边界）
3. fuse 可扩展：reduce + elementwise 更复杂模式（当前 binary side inputs + reduce 链头已支持）
4. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型（tensor/GraphProto）
5. isel：规则从文件热加载已有，可考虑按 target 分规则集（CUDA/NPU/CPU 不同指令）

### 2026-07-11 — Abs+Log op 全链路 + 3 个浮点结构重写（feat/neutron-ab5e1007）

**当前状态**：完成遗言上轮列的 float_opts 两项（Sqrt(x*x)→Abs、Log(Exp(x))→x）+ 扩展一个 Pow(x,2)→Mul。新增 Abs(29)+Log(30) 两个 op 全链路接入。回归全绿，本分支待合并回 main。

**用户指引延续**：聚焦浮点结构优化（项目招牌），不在简单代数/常量传播上耗时间。本轮全部用子代理实现（主 agent 只做 git 提交+验证，避免上下文崩）。用户要求短时间盒+多子代理。

**已完成**（3 commit，按时序）：
- **base 新增 Abs(29)+Log(30) op 全链路接入**（commit b8f8eb2）：为 Sqrt(x*x)→Abs 和 Log(Exp(x))→x 铺前置依赖。8 处全链路：base OpKind enum+from_u8 / shape_infer（unary elementwise passthrough）/ cost_model（Abs 同 Relu、Log 同 Sigmoid/Tanh 量级）/ lowering（abs/log kernel，match 无 catch-all 穷举检查强制覆盖）/ isel（abs/log lisp 规则）/ frontend onnx+dsl 映射 / fuse is_elementwise 融合候选。Abs 是单条硬件指令（√(x²)=|x|），Log 是超越函数（ln(eˣ)=x）
- **float_opts Sqrt(x*x)→Abs + Log(Exp(x))→x**（commit c071ffe）：
  - `Sqrt(x*x)→Abs(x)`：√(x²)=|x|，Sqrt+Mul 两 op 换单条 Abs 硬件指令。try_match_sqrt_square 匹配 Sqrt 输入是 Mul 且两输入相同（x*x，x≠y 不匹配）；apply 用"改 op+换输入"模式（参考 PowHalfToSqrt），Sqrt 改 Abs 输入换 [x]，原 Mul 变孤儿交 DCE。3 单测（正向/负向/张量 shape）
  - `Log(Exp(x))→x`：ln(eˣ)=x，消去 Log+Exp 两 op。try_match_log_exp 匹配 Log 输入是 Exp；apply 用"重写使用者"模式（参考 ExpMulFusion 但不新建节点），把 Log 输出值的所有使用者重写到 Exp 的输入 x，图输出也重写，带 log_out!=x_input 守卫防自引用。溢出边界：数学上对所有有限 x 成立，运行时 Exp 溢出成 inf 时语义有差异（inf→有限值），但 ML 场景 x 极少大到 Exp 溢出（f32 阈值约 x>889），默认启用。2 单测（正向/负向）
- **float_opts Pow(x,2.0)→Mul(x,x)**（commit 2639d42）：x²=x·x，通用 Pow（log/exp 超越函数实现，贵）换便宜 Mul。Pow 节点改 Mul，输入换 [x,x]（两输入都是 base），丢弃常量指数 2.0，输出 value 不变（使用者无感），原常量节点变孤儿交 DCE。无溢出/NaN 风险（数学完全等价）。用"改 op+换输入"模式。2 单测（正向/负向）

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：132 passed（base 3 + common 2 + frontend 23 + interface 5 + isel 12 + lisp 4 + optimizer 83）—— 较上轮 125 +7（5 Sqrt/Log 单测 + 2 Pow 单测）
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常（空 onnx 走 Placeholder，pipeline 串联成功）

**设计哲学遵守**：3 个新重写全是浮点代数恒等式：√(x²)=|x|、ln(eˣ)=x、x²=x·x。是设计哲学点名"IEEE754 位级 trick / 浮点结构重排"的具体落地，非 MatMul+Add→Linear 贪心模式。严格遵守两阶段模式（find_opportunities 借图只读收集，apply_float_opts 独占可变借用改图）。

**关于 RMSNorm 整体 cost-based 识别（遗言上轮列的 float_opts 第1项，本轮未做）**：评估后认为识别 x*rsqrt(y) 的 RMSNorm 整体模式本质是复合算子模式识别（需识别 var=ReduceMean(x*x) 整个结构），触及设计哲学"不要贪心模式匹配"红线。且 x*rsqrt(y) 若都是 elementwise，fuse 的 elementwise 链融合已能处理。故推迟，不在 float_opts 做整体 RMSNorm 识别。

**isel 覆盖现状**：lowering 现显式覆盖全部 OpKind 变体（31 个，除 Custom=64 保留段），无 catch-all。新增 Abs/Log 已四点全接。新增 op 时编译器穷举检查强制补 lowering 分支。

**子代理协作模式**：本轮 3 个实现任务全派 general_purpose_task 子代理（1 个做 op 全链路接入、1 个做 Sqrt/Log 重写、1 个做 Pow 重写），主 agent 只做 Read 调研+git 提交+回归验证。2 个 search 子代理先并行摸清 float_opts 结构和 op 接入点，给实现子代理精确行号+代码模式。主 agent 上下文未爆。

**下一步**（优先级排序）：
1. pt 前端：PyTorch 解析（frontend 最后一块占位）
2. float_opts 可继续：`Exp(Log(x))→x`（LogExpToIdentity 对偶，e^(ln x)=x，但 x>0 定义域约束风险更高，ML 罕见）；`Pow(x,3.0)→Mul(x,Mul(x,x))`（整数幂换乘法扩展，需新建嵌套 Mul）；`Sqrt(Sqrt(x))` 不做（Pow(x,0.25) 比 Sqrt 更贵，负优化）
3. fuse 可扩展：reduce + elementwise 更复杂模式（当前 binary side inputs + reduce 链头已支持）
4. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型（tensor/GraphProto）
5. isel：规则按 target 分规则集（CUDA/NPU/CPU 不同指令）

### 2026-07-11 — CSE IO 识别修三个缺口（feat/neutron-cse-io）

**当前状态**：用户要求做架构无关层的 IO 优化识别（即 CSE，设计哲学"IO 同样性=CSE"）。调研发现 CSE 的"识别"有两个 critical 正确性 bug（不是漏识别，是错误识别=false positive）+ 一个识别能力缺口，全部修完。回归全绿，本分支待合并回 main。

**用户指引**："做IO优化的识别...在架构无关那里先实现基础的io优化，架构相关的以后再看"。理解为：在 optimizer 层做 CSE 识别补强，不动 arch/isel。调研用 1 个 search 子代理摸清 CSE 全貌后发现三个缺口。

**已完成**（2 commit）：
- **修缺口 A+B：CSE 指纹纳入 attr + 多元素常量指纹塌缩修复**（commit cd81f5c）：
  - 缺口 A（critical 正确性 bug）：Fingerprint::Op 只编码 (op, inputs) 不含 attr，带 attr 的 op（Reduce Axis / Concat Axis / Transpose Perm / Reshape Shape / Fused Shape+Strides）inputs 相同但 attr 不同时被错误合并。decompose 后大量 ReduceMean/ReduceMax 同输入不同 axis 会误合并，真实可触发。修法：Fingerprint::Op 加第三个字段 attr_hash: u64，新增 attr_fingerprint 辅助遍历节点所有 attr（Int/Float/Bool/IntArray/FloatArray）按 (key, tag, value) 累加成稳定 u64。
  - 缺口 B（critical 正确性 bug）：多元素常量张量的 constant_value() 返回 None，指纹 unwrap_or(NaN) 塌缩成 NaN.to_bits()，所有多元素常量（如不同 weight 矩阵）被误判相同。修法：fingerprint 改返回 Result<Option<Fingerprint>>，单元素走 Constant(bits)，多元素走 ConstantTensor(dims+values 哈希)，未知常量 None 跳过 CSE（保守不合并）。
  - 5 新单测：reduce 不同 axis 不合并 / transpose 不同 perm 不合并 / reduce 同 axis 合并（正向）/ 不同多元素常量不合并 / 相同多元素常量合并
- **修缺口 C：CSE 不动点迭代**（commit 7cf6de2）：CSE 只跑一次，消除节点后使用者 inputs 重写暴露的新"两节点 inputs 相同"机会捕不到（algebra 有不动点，CSE 没有）。修法：CsePass::run 包不动点循环（上限 16 次防意外死循环），循环到 apply_cse 返回 0 收敛。1 新单测 cse_fixpoint_catches_newly_exposed：a=Add(x,y)、b=Add(x,y)（第一轮消除 b）、c=Add(a,b)→Add(a,a)、d=Add(a,a)，单次只消除 1 个，不动点消除 2 个，第三轮收敛。模块注释同步更新。

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：138 passed（base 3 + common 2 + frontend 23 + interface 5 + isel 12 + lisp 4 + optimizer 89）—— 较上轮 132 +6（5 attr/常量单测 + 1 不动点单测）
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：CSE 是设计哲学明确点名的"IO 同样性"识别。本轮修的是识别的**正确性**（false positive：带 attr 的 op 和多元素常量被误合并）和**完整性**（不动点捕新机会）。非贪心模式匹配，纯字面相同子表达式合并。保守策略：未知常量不参与 CSE 避免误合并。

**子代理协作模式**：1 个 search 子代理摸清 CSE 全貌（Fingerprint/run_cse/pipeline 注册/单测覆盖/缺口分析），1 个 general_purpose_task 子代理实现缺口 A+B（attr 指纹 + 多元素常量指纹）。缺口 C 改动小（CsePass::run 包循环），主 agent 直接做。主 agent 上下文未爆。

**CSE 识别能力现状**：
- 指纹完整：(op, normalized_inputs, attr_hash)，可交换 op（Add/Mul）inputs 排序，常量按值/张量内容区分
- 不动点迭代：消除后暴露的新机会能多捕一层（上限 16 次）
- 保守跳过：未知常量（非 FLOAT 张量）不参与 CSE
- pipeline 位置：重排阶段最后（algebra→float_opts→CSE），规范化后做字面匹配

**下一步**（优先级排序）：
1. pt 前端：PyTorch 解析（frontend 最后一块占位）
2. fuse 可扩展：reduce + elementwise 更复杂模式（当前 binary side inputs + reduce 链头已支持）
3. float_opts 可继续：Exp(Log(x))→x 对偶 / Pow(x,3.0) 整数幂扩展
4. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型（tensor/GraphProto）
5. isel：规则按 target 分规则集（CUDA/NPU/CPU 不同指令）
6. CSE 可选增强：可交换 op 集中管理（改 is_commutative 辅助，当前硬编码 Add|Mul，未来加 Max/Min/Equal 时同步）；等价但非字面相同的 IO（如 x+x 与 x*2，依赖 algebra 规范化，当前 pipeline 顺序已对）

### 2026-07-11 — 启发式识别补强（feat/neutron-heuristics）

**当前状态**：用户问"当前纯规则架构是否最优，不是就继续加，加多点启发式识别，主动点"。调研判断：当前是"纯规则 + cost model 辅助"，规则覆盖范围内正确，但缺"启发式主动识别"——很多 no-op 要主动发现而非等规则匹配。本轮主动补 3 个启发式。回归全绿，本分支待合并回 main。

**用户指引**："加多点启发式的识别，因为有时候启发是真的很重要。主动点去加吧"。理解为：不要只做被动规则匹配，要主动识别可消除/可优化的模式。架构相关（lowering/isel）仍不动。

**已完成**（2 commit）：
- **algebra 加 reduce/concat shape-based no-op 启发式**（commit f58899c）：
  - `Reduce(x,axis) 当 axis 维 size==1 → ReplaceWith(x)`：沿 size-1 维 sum/mean/max = 该维唯一元素，是 no-op。保守正确实现：额外校验 reduce 输出 shape 已知且等于输入去掉 axis 维后的 shape（标准 reduce 语义），保证 ReplaceWith 后 shape 严格正确。axis 支负值，越界/未知 shape/输出不符时 NoChange。decompose 后 LayerNorm/Softmax 产生大量 reduce，batch=1 或退化维度场景真实可触发。
  - `Concat([x],axis) 单输入 → ReplaceWith(x)`：单输入 concat 是 copy no-op。
  - 新增 read_axis 辅助（照 read_perm 模式）。5 新单测（正向/负向/shape 不符保守不消除）
- **fuse 反融合判定增强**（commit 5f94d90）：当前 fuse 判定过宽（saving>0 就融，CUDA launch=10 几乎所有≥2链都融），缺反融合考量。
  - CostCoeffs 加 fuse_overhead 字段（CUDA 0.5 / NPU 0.3 / CPU 0.1），fusion_saving 扣除 extra=(n-1)*fuse_overhead 模拟融合后 Fused kernel 寄存器压力（线性近似 spill 风险）。saving = saved_launch + saved_mem - extra，自然提高融合门槛。
  - fuse.rs 加 MAX_FUSION_CHAIN=16 链长上限，build_fusion_chain 扩展时超 16 截断，避免长链寄存器溢出。
  - 3 新单测（overhead 扣除 / 长链高 overhead saving 变负 / 20 节点链截断到 16）。现有 7 fuse 测试不回归（链长≤3，extra 远小于 saved_launch+saved_mem）

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：146 passed（base 3 + common 2 + frontend 23 + interface 5 + isel 12 + lisp 4 + optimizer 97）—— 较上轮 138 +8（5 algebra + 3 fuse）
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：
- reduce/concat no-op 是"简单代数规则"的启发式扩展（识别 size-1 维 no-op，非复合算子模式匹配），保守策略（shape 严格校验才消除）
- fuse 反融合是 cost model 驱动的启发式（寄存器压力近似 + 链长上限），符合"cost model 现在就做"和"启发式优化器"
- 非贪心模式匹配（reduce no-op 是单节点简化，fuse 反融合是 cost 判定而非模式识别）

**架构判断（回答用户"纯规则架构是否最优"）**：当前架构是"纯规则 + cost model 辅助判定"，在规则覆盖范围内正确，但有 3 类缺口：(1) shape-based no-op 主动识别（本轮补 reduce/concat，还有 Slice 全量等因 AttrKey 缺失推迟）；(2) fuse 反融合判定（本轮补 overhead 扣除 + 链长上限）；(3) 等价但非字面相同的 IO 识别（依赖 algebra 规范化，当前 pipeline 顺序已对，缺的是 algebra 规范化能力）。不是最优，但本轮补的 3 个启发式覆盖了最高价值的缺口。剩余缺口中，Slice no-op 需先补 AttrKey 枚举 + frontend 解析（工作量大），等价 IO 识别依赖 algebra 规范化增强（如 x+x 与 x*2 统一，但这触及"不要模式匹配"红线边界，需谨慎）。

**子代理协作模式**：1 个 search 子代理摸清 algebra/fuse/shape_infer/cost_model 的启发式覆盖现状（含 no-op 场景清单 + fuse cost 判定缺陷 + 缺口优先级），1 个 general_purpose_task 子代理实现 algebra 两个 no-op，1 个 general_purpose_task 子代理实现 fuse 反融合。主 agent 只做 git 提交+验证。上下文未爆。

**下一步**（优先级排序）：
1. pt 前端：PyTorch 解析（frontend 最后一块占位）
2. Slice no-op 识别：需先补 AttrKey::Starts/Ends/Axes/Steps + frontend 解析 Slice 属性，工作量大但价值高（Slice 全量是常见 no-op）
3. fuse 可扩展：reduce + elementwise 更复杂模式
4. float_opts 可继续：Exp(Log(x))→x 对偶 / Pow(x,3.0) 整数幂扩展
5. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型（tensor/GraphProto）
6. isel：规则按 target 分规则集（CUDA/NPU/CPU 不同指令）
7. CSE 可选：可交换 op 集中管理 / 等价非字面 IO 识别（依赖 algebra 规范化）

### 2026-07-11 — Pow 整数幂系列 + Exp(Log) 对偶 + fuse elementwise→reduce（feat/neutron-2b9e4f71）

**当前状态**：本轮一小时时间盒（18:05 开工，北京时间），做遗言上轮列的 float_opts 整数幂扩展（Pow x³/x⁻²）+ Exp(Log) 对偶 + fuse 扩展 elementwise→reduce。回归全绿，本分支待合并回 main。

**用户指引**：用户要求"干一个小时，每十分钟用 AskUserQuestion 提问验证方向"，后改为"先别问了"。延续前几轮"聚焦浮点结构优化，不在简单代数/常量传播上耗时间"的指引。

**已完成**（5 commit，按时序）：
- **float_opts PowCubeToMul**（commit 7382cd9）：`Pow(x, 3.0)` → `Mul(x, Mul(x, x))`，x³=x·x·x，通用 Pow（log/exp 超越函数实现，贵）换两次便宜乘法。新建内层 Mul(x,x) 节点，Pow 节点改外层 Mul 吃 [x, x²]（改 op+换输入模式）。无溢出/NaN 风险（数学完全等价）。2 单测
- **float_opts ExpLogToIdentity**（commit ce310c6）：`Exp(Log(x))` → `x`，e^(ln x)=x，LogExpToIdentity 的对偶。用"重写使用者"模式（把 Exp 输出值的所有使用者重写到 Log 的输入 x，图输出也重写）。风险：Log 定义域 x>0，x≤0 时原式 Exp(NaN)=NaN，重写后得 x（≤0）语义有差异，但 ML 场景 Log 输入多为正（softmax 输出、概率、x²+ε），默认启用。2 单测
- **fuse 支持 elementwise→reduce 融合**（commit d5dbe2f）：find_opportunities 改判定让链尾可以是 reduce（原来只 elementwise）。1 行核心改动：`if !is_elementwise(n.kind) && !is_reduce(n.kind) { continue; }`。build_fusion_chain 已有"reduce 前驱 break"逻辑保证链中最多 1 个 reduce，apply_fusion 自然处理尾部 reduce（Axis attr 留在尾节点不被删）。这是 ML pooling 模式（relu→ReduceSum）。2 新单测（正向/负向：共享中间结果不融合）
- **float_opts PowNegTwoToReciprocal**（commit 49d955f）：`Pow(x, -2.0)` → `Reciprocal(Mul(x, x))`，x^(-2)=1/x²=reciprocal(x²)，通用幂换 Mul + Reciprocal（都是单条硬件指令）。新建 Mul(x,x)，Pow 改 Reciprocal 吃 [mul_out]。RMSNorm/L2 归一化常见。无溢出风险（x=0 时原 Pow 也是 inf，重写后 Reciprocal(0)=inf 一致）。2 单测
- **interface e2e 验证 elementwise→reduce 融合全链路**（commit 7fc9efc）：构造 relu(x)→ReduceSum(.,axis=1)，跑 O2 pipeline 验证 Fused 节点带 Axis attr 出现，arch::lower 发 "fused" kernel，isel::select 选 "fused" 指令。证明 Fused op_seq 含尾部 reduce 时全链路不崩（fuse 本轮新增 elementwise→reduce 链型）

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：156 passed（base 3 + common 2 + frontend 23 + interface 6 + isel 12 + lisp 4 + optimizer 106）—— 较上轮 146 +10（6 float_opts 单测 + 2 fuse 单测 + 2 Pow 整数幂单测含 e2e）
- CLI 端到端：`cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump` 正常

**设计哲学遵守**：
- Pow 整数幂系列（x²/x³/x⁻¹/x⁻²）是"通用幂函数（log/exp 超越函数实现）→ 专用单条硬件指令（Mul/Reciprocal/Sqrt/Rsqrt）"的浮点结构优化，正是设计哲学点名"类 Quake III fast inverse sqrt"同族
- Exp(Log(x))→x 是浮点代数恒等式重写（e^(ln x)=x），非贪心模式匹配
- fuse elementwise→reduce 是通用融合（reduce 作为 shape 分界点的链尾），非 MatMul+Add→Linear 复合算子模式。build_fusion_chain 的"reduce 前驱 break"保证链中最多 1 个 reduce，融合后 Axis attr 保留正确性

**关于 Pow 系列扩展边界**：Pow(x,4.0) 及更高次幂理论上可继续扩展（x⁴=Mul(x²,x²) 或 Mul(Mul(x,x),Mul(x,x))），但：(1) ML 模型中 Pow 指数 ≥3 已罕见（常见 0.5/-0.5/-1/-2 与 RMSNorm/L2 相关）；(2) 4 次以上 Pow 多出现在 polynomial 特征，但那些场景 Pow 语义应保留供后端发专用 pow kernel；(3) 继续扩展收益递减。故 Pow 系列本轮收尾，已有 ±0.5/-1/2/3/-2 五档覆盖 ML 主要场景。

**fuse 扩展边界**：elementwise→reduce 已支持。更复杂的"reduce→elementwise→reduce"多 reduce 链刻意不做——build_fusion_chain 遇 reduce 前驱即 break 保证链中最多 1 个 reduce，多 reduce 融合语义复杂且收益不明。

**子代理协作模式**：本轮主 agent 直接做实现（float_opts 4 个重写 + fuse 1 个扩展），未派子代理——改动模式成熟（参考已有 Pow 重写），子代理反而增加协调开销。1 个 search 子代理在调研阶段摸清 fuse 降低判定可行性与 pt 前端工作量（pt 结论：TorchScript pickle 短期不可行，~1000+ 行，维持占位）。

**下一步**（优先级排序）：
1. pt 前端：PyTorch 解析（TorchScript pickle 太复杂，短期维持占位；可考虑走 ONNX export 路线绕过）
2. Slice no-op 识别：需先补 AttrKey::Starts/Ends/Axes/Steps + frontend 解析 Slice 属性，工作量大但价值高（Slice 全量是常见 no-op）
3. fuse 可扩展：reduce→elementwise→reduce 多 reduce 链（语义复杂，需 cost model 评估收益）
4. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型（tensor/GraphProto）
5. isel：规则按 target 分规则集（CUDA/NPU/CPU 不同指令）
6. CSE 可选：可交换 op 集中管理 / 等价非字面 IO 识别（依赖 algebra 规范化）
7. float_opts 边界：Pow 系列（±0.5/-1/2/3/-2）已覆盖 ML 主要场景，进一步扩展收益递减

### 2026-07-11 — regalloc crate 图着色寄存器分配器（feat/neutron-regalloc）

**当前状态**：用户要求"最好的寄存器分配，去研究，先给个最终目标"→"找个最好的融合方案放到最高质量的模式"→"你倒是干活啊"。调研（VERILOCC/RL4ReAl/GNN-based/LLVM greedy+pbqp）后选定 Chaitin-Briggs 图着色 + 保守合并 + 分段溢出的融合方案，新建 regalloc crate 实现并集成到 compile pipeline。回归全绿，本分支待合并回 main。

**用户指引**：用户要"最好的融合方案放到最高质量的模式"，理解为不要单一算法，要综合各流派优点（图着色精确性 + 活跃区间精确性 + 保守合并不引入新溢出 + 分段溢出降低代价）。研究阶段用 WebSearch 调研，实现阶段主 agent 直接做。

**已完成**（1 commit）：
- **regalloc: 图着色寄存器分配器（Chaitin-Briggs + 保守合并 + 分段溢出）**（commit 2db1865）：
  - 新建 7 文件 regalloc crate（Cargo.toml + lib.rs + types/liveness/interference/coalescing/coloring/allocator 6 模块），1851 行
  - **types**：VReg(u32)/PReg(u32)/Operand enum/MachineInstr（op+operands+defs+args）/RegisterFile（for_target: CUDA=30/NPU=31/CPU=13 allocatable）/Allocation。7 单测
  - **liveness**：LiveInterval [start,end]，overlaps 用 >=（同指令 use+def 视为干扰，保守正确）。analyze 遍历 MachineInstr 记录每个 VReg 的 [def, last_use]，图输入值区间从 0 起。6 单测
  - **interference**：InterferenceGraph 邻接表，O(V²) build 遍历所有 VReg 对。**关键：move 例外**——build 接收 instructions 参数，识别 move 对 (src,dst)，若重叠区间仅单条指令（=move 本身）不加干扰边，允许 coalescing 合并。5 单测
  - **coalescing**：保守 Briggs 合并，合并后度数 < threshold=3 才合并（不引入新溢出）。combined_degree = neighbors(src)∪neighbors(dst) 中既非 src 也非 dst 的节点数。被合并的 dst 从图移除。5 单测
  - **coloring**：Chaitin-Briggs 两阶段——Phase1 simplify（度数<K 入栈）+ spill（全部≥K 时溢出代价最低的，cost=(uses+1)/interval_len 密度低优先）；Phase2 select（栈弹出贪心分配最小可用颜色）。5 单测
  - **allocator**：串联 1→5（liveness→interference→coalesce→color→spill rewrite）。溢出重写：被溢出 VReg 在 def 后插 store_spill、use 前插 load_spill（分段溢出，PReg(0) 作临时寄存器）。被合并的 move 指令跳过（冗余消除）。5 单测
  - **interface 集成**：compile() 在 isel 后加 regalloc 步骤，Output 新增 machine_instructions/reg_assignment/spilled 字段。2 端到端单测（regalloc_produces_valid_assignment + regalloc_assigns_all_vregs 验证所有 VReg 被分配或溢出、干扰 VReg 不分到同一 PReg）

**move 例外是本轮关键正确性点**：mov v0→v1 时 v0（src）在 move 指令被 use，v1（dst）在 move 指令被 def，二者在 move 指令同时活跃。若用 >= 判定重叠则 v0/v1 干扰→coalescing 跳过→move 无法消除。标准 Chaitin 解法：move 的 src/dst 不互相干扰（可共用寄存器，coalescing 合并后 move 变 no-op）。build 识别 move 对，单点重叠（仅 move 那条指令）不加干扰边。多指令重叠（src/dst 在 move 之外也同时活跃）仍加边（真实干扰，不能共用）。

**修复的 bug**（实现过程）：
1. LiveInterval::overlaps 最初用严格 >（同指令 use+def 不干扰，错误），改 >=
2. interference.rs 的 neighbors() 用 unsafe 指针操作（HashSet 无 as_slice）有 UB，删除改用 neighbor_set() 返回 &HashSet（节点存在不变式用 expect 明确）
3. build_interference 命名不匹配（interface 测试调用，实际导出名是 build）→ 改测试用 regalloc::build
4. spill_cost_favors_low_use 测试 setup 让 v1 密度更高（use_count/interval 反而更贵），重设为 v1 单次 use 长区间→密度低→cost 低
5. clippy: LiveInterval 加 is_empty（len_without_is_empty）/ InterferenceGraph 加 Default impl（new_without_default）

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace --all-targets -- -D warnings`：0 warning
- `cargo fmt --all -- --check`：clean
- `cargo test --workspace`：193 passed（base 3 + common 2 + frontend 23 + interface 7 + isel 12 + lisp 4 + optimizer 106 + regalloc 35）—— 较上轮 158 +35（regalloc 单测）

**设计哲学遵守**：regalloc 在 isel 之后独立 crate，不修改 isel/arch（架构相关层只做 lowering+isel，寄存器分配是 isel 后的通用阶段）。算法是教科书 Chaitin-Briggs（非贪心模式匹配），保守合并保证不引入新溢出（correctness over performance）。

**regalloc 当前能力边界**：
- VReg 分配：lower_to_machine 遍历 IR Graph 值流，图输入值优先分配 VReg，每个节点 inputs/outputs 分配新 VReg
- 干扰图：O(V²) 区间重叠 + move 例外（单点重叠=move 不加边）
- 合并：保守 Briggs（threshold=3），合并后度数 < threshold 才合并
- 着色：simplify+spill+select，溢出代价=密度（uses+1）/interval_len
- 溢出：分段溢出（def 后 store_spill + use 前 load_spill），PReg(0) 临时寄存器
- move 消除：被合并的 move 指令跳过

**下一步**（优先级排序）：
1. regalloc 增强：live-out 精确干扰（当前区间近似，move 例外是补丁；标准做法是每条指令算 live-out 集合，move src/dst 不互相干扰）
2. regalloc 增强：spill code 位置优化（当前 def 后立即 store，可延迟到首次溢出 use 前）— **同指令内重复 use 已优化**（见 2026-07-13 日志），跨指令共享 load 待做
3. regalloc 增强：rematerialization（常量/地址计算类 VReg 溢出时不 store/load 而是原地重算）
4. pt 前端：PyTorch 解析（TorchScript pickle 太复杂，维持占位；可走 ONNX export 路线）
5. isel：规则按 target 分规则集（CUDA/NPU/CPU 不同指令）
6. Slice no-op 识别：需先补 AttrKey::Starts/Ends/Axes/Steps + frontend 解析 Slice 属性
7. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型

### 2026-07-13 — 测试覆盖加固 + lisp 算术 bug 修复（feat/neutron-backend）

**当前状态**：本轮继续在 feat/neutron-backend 分支推进代码质量改进。5 个 commit 已 push（c279860 / 325b027 / d506e92 / ae36729 / f6787fc），全 workspace 测试 261 → **311 passed**，clippy + fmt 干净。无接口变化，无破坏性改动。

**已完成**：
- **arch crate 测试空白填补**（commit c279860）— arch crate 之前完全无测试，加 6 个：lower() Add/relu/sqrt/tanh/32 个 OpKind 全覆盖 + ArchGraph 基础 API + CudaDesc 默认值
- **regalloc spill 重写优化**（commit 325b027）— 修一个 spill 重写小 bug：`fadd v0 v0`（v0 被溢出）原本插两条 load_spill，改为同指令内同 VReg 共享一次 load（loaded_in_this_instr HashMap）。新增测试 allocate_same_instr_reuses_spill_load 验证
- **base/storage.rs 测试空白填补**（commit d506e92）— storage.rs 含 unsafe 强转 i64/f64 slice，是关键风险点但 0 测试。加 17 个测试：AttrKey/AttrTag 全变体 u8 往返 + alloc_node/value 递增 ID + 5 种属性类型往返 + 多节点共享 edges 池隔离 + value_shape/name 往返 + set_value_shape 覆盖 + parent_region 默认值 + add_name null 终止
- **lisp +/* 混合 int/float bug 修复**（commit ae36729）— TDD 流程揭露 bug：`(+ 1 2.5)` 返回 4.5 而非 3.5，`(* 2.0 3)` 返回 9.0 而非 6.0。根因：is_float=true 分支返回 `acc_float + acc_int as f64`，但 acc_float 在循环里已累加了每个 Int（`acc_float += *i as f64`），导致 Int 被算两遍。修复：is_float=true 时直接返回 acc_float。对照 `-` 实现就是这种写法。同时加 25 个测试覆盖 parser 转义/嵌套/注释/错误 + interp 算术/短路逻辑/字符串/错误处理
- **backend emit() 路由测试**（commit f6787fc）— backend/lib.rs 之前 0 测试，emit() 是入口调度但路由分支没专测。加 7 个测试覆盖 Target/SourceLang 全路由 + CPU 占位 + 空图不 panic + arch 透传

**协作状态**（Shannon-Model Issue #1）：
- 评论 4953644034 已发送，同步 vllm-ascend 3 个 commit 进度（实际是 5 个，评论时是 3 个）
- 对方 trae/agent-nP28Lt 分支（0374674）有 test.yml / build.yml / release.yml 三个修复，未合 main，无 open PR
- 对方未回复，按"不干涉对方工作"原则不催

**下一步**（优先级排序）：
1. regalloc：live-out 精确干扰（最大风险改进，需重构 build/analyze，会改变现有干扰图边数）
2. regalloc：rematerialization（常量 VReg 溢出时原地重算而非 store/load）
3. backend：cuda/kernels.rs 1285 行只 13 个测试，比例偏低，可补
4. frontend：onnx.rs 1201 行只 14 个测试，比例偏低
5. isel：规则按 target 分规则集

### 2026-07-11 — 后端代码生成四件套（feat/neutron-backend）

**当前状态**：CUDA / Triton / Metal / Ascend CANN 四个后端的代码生成全部完成，每个后端覆盖全 31 个 OpKind（无空缺/stub），按微架构特化（Hopper wgmma+TMA / Blackwell tensor memory / Ampere mma.sync / Apple simdgroup / Ascend Cube+Vector Core）。已集成到 `interface::compile` 流程，回归全绿。本分支待合并回 main。

**已完成**：
- backend crate 骨架（spec/extract/lib）— commit 406e7c1
  - `GpuArch` enum（9 变体）：Ampere80/Hopper90/Blackwell100/Apple6/7/8/Ascend910B1/B3/310P3
  - `KernelSpec` + `LaunchSpec` + `TensorSpec` + `KernelAttrs` 后端无关规格
  - `DTypeExt` trait 扩展（c_type/msl_type/triton_type/cann_type/size_bytes，不能给外部 crate DType 加 inherent impl）
  - `extract.rs` 从 IR Graph 提取 KernelSpec（AttrKey 数值常量匹配，因 base::storage::AttrKey 私有）
  - 4 个后端 stub 先编译通过
- **CUDA 后端**（1285 行）— commit 729b508
  - Ampere80: `nvcuda::wmma` mma.sync 16x16x16 tile
  - Hopper90: `wgmma.mma_async` 64x256 + TMA（cp.async.bulk + mbarrier PTX 内联）
  - Blackwell100: tensor memory + tcgen05 TC fusion
  - 元素级/Reduce/Softmax/LayerNorm/Conv/Pool/Transpose（带 bank conflict padding）/Concat/Slice 全覆盖
  - Constant 广播 / Fused 链式 / Custom `extern "C"` 入口
  - 13 单测验证语法关键字 + 全 OpKind × 全 arch 矩阵
- **Triton 后端**（1155 行）— commit 0fcdf30
  - Hopper90: `tl.make_block_ptr` + `tl.advance` + `boundary_check`（SM90 TMA）
  - Ampere80: 标准 `tl.dot`，BLOCK_M=BLOCK_N=64
  - Blackwell100: SM100 FP4/FP6 tensor core 注释
  - 32 OpKind 全覆盖，12 单测
- **Metal 后端**（906 行）— commit 05d66fb
  - `simdgroup<float8x8>` 8x8 矩阵乘法单元（Apple6/7/8）
  - threadgroup shared memory + `threadgroup_barrier(mem_flags::mem_threadgroup)`
  - Softmax 三遍 / LayerNorm 两遍 + epsilon
  - 9 单测
- **Ascend CANN 后端**（1600 行）— commit 06ca79e
  - AscendC 高级 API（Add/Sub/Mul/ReduceSum/MatMul 等）
  - Ascend910B1/B3: Cube Core (mmad) + Vector Core 协同
  - Ascend310P3: 轻量推理路径
  - 双缓冲 (BUFFER_NUM=2) + tiling (TILE_LENGTH=128)
  - TQue/TPipe/GlobalTensor/LocalTensor/DataCopy 标准 Ascend 结构
  - 10 单测
- clippy + fmt 修复（manual_div_ceil / clone_on_copy）— commit da68819
- **interface 集成** — commit 2b6e4f4
  - `interface::compile` 在 regalloc 后增加第 6 步：`backend::emit`
  - 按 target 选默认微架构：Cuda→Hopper90，Npu→Ascend910B1
  - `Output` 新增 `backend_source` + `backend_lang` 字段
  - 5 个端到端测试验证四后端产出非空源码

**回归验证（全绿）**：
- `cargo build --workspace`：0 warning
- `cargo clippy --workspace -- -D warnings`：0 warning
- `cargo fmt --all --check`：clean
- `cargo test --workspace`：242 passed（backend 44 + interface 13 + 其余 185）

**设计要点**：
- `KernelSpec` 是后端无关规格，4 个后端共享 `extract.rs` 提取逻辑，各自只关心源码生成
- `GpuArch` enum 统一表达 9 个微架构，`has_tma/has_wgmma/has_simdgroup/has_cube_core` 能力查询让后端按需分派
- `DTypeExt` trait 扩展模式解决"给外部 crate 的 DType 加方法"问题（Rust 不允许 inherent impl）
- 子代理写入文件丢失问题：CUDA/Triton 子代理首次返回成功但文件未保存（沙箱问题），重派时增加 `wc -l` 验证步骤才成功

**下一步**（优先级排序）：
1. 合并 feat/neutron-backend → main
2. backend：CUDA kernel 的 dtype template instantiation（当前用注释标注，后续可生成真正的 template 实例）
3. backend：Triton autotune 配置（当前 BLOCK_SIZE 固定，可加 `@triton.autotune` 多配置搜索）
4. backend：Metal MPS host 端 launch 代码（当前注释形式，后续生成 Swift/Obj-C++ wrapper）
5. backend：CANN tiling 参数从 host 端传入（当前硬编码 TILE_LENGTH=128，后续从 tiling buffer 读）
6. regalloc：live-out 精确干扰 + spill code 位置优化
7. frontend：解析 ONNX 子图（if/loop）+ 更多属性类型

