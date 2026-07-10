# WORKSPEC.md — 新 agent 上手标准流程（work-spec）

> 每次新会话开始，agent 必须按此文件执行。用户不想每次重复讲。
> 本文件与 AGENTS.md 配合：AGENTS.md 是项目宪法（设计哲学 + 防 P0 规则 + Continuity Log 遗言），本文件是操作 checklist + **时间盒执行规则**。

## 0. 必读（动手前）
- 通读 `AGENTS.md`，重点看末尾 **Continuity Log 最后一条遗言**：了解前任做到哪、下一步优先级是什么
- 通读本文件
- 不确定的设计决策，先问用户，不要擅自变更设计哲学

## 1. 验证回归（动手前必做）
确认环境正常、主干没被破坏。四件套必须全绿：

```bash
cargo build --workspace                              # 0 warning
cargo clippy --workspace --all-targets -- -D warnings  # 0 warning
cargo fmt --all -- --check                           # clean
cargo test --workspace                               # 全绿
```

任一不过：先报告用户，**不要在坏的基础上开工**。

## 2. 建新分支
从 `main` 切出，命名 `feat/neutron-<8hex>`（公共前缀 + 哈希，能复用就复用，不建一堆分支）：

```bash
git checkout main && git checkout -b feat/neutron-<8hex>
```

## 3. 时间盒执行（核心规则）
**用户会规定一个工作时长（如 30 分钟）。agent 必须做满这个时长，没到点不许结束。**

### 3.1 开工先计时（转北京时间 UTC+8）
建完分支、回归确认后，第一件事用工具算时间：

```bash
# 开始时间（北京时间）
TZ='Asia/Shanghai' date '+开始: %Y-%m-%d %H:%M:%S %Z'
# 预计结束时间 = 开始 + 用户规定时长（例 30 分钟）
TZ='Asia/Shanghai' date -d '+30 minutes' '+预计结束: %Y-%m-%d %H:%M:%S %Z'
```

把这两个时间记下来（写在回复里 + 记在心里）。**结束时间一到才允许收尾**。

> 注：本环境是 Linux，`date -d '+N minutes'` 是 GNU date，可用。若该命令不可用，用 `date -u` 拿 UTC 再手动 +8 转北京 + 手算结束。

### 3.2 持续列待办 → 完成待办 → 再列待办（循环到时间到）
在规定时长内，**不许空闲、不许提前结束**。循环模式：

1. **列待办**：用 `TodoWrite` 把下一步能做的事拆成 3-6 个待办项（从遗言"下一步"优先级取）。每个待办是一个**独立可提交单元**（一个 pass / 一个修复 / 一组测试）
2. **完成待办**：逐个做，做完一个立刻 `git commit -s` + 跑 `cargo build && cargo test` 确认没崩
3. 待办清空后，**回到第 1 步再列新待办**（从遗言下一步、或刚做时发现的新缺口取）
4. 每完成一个待办，用 `TZ='Asia/Shanghai' date` 看一眼当前时间，判断是否到点
5. **到点才进入第 5 步收尾**（写遗言 + 合并）

### 3.3 多用子代理并行（提速关键）
为在规定时间内多做任务，**尽可能用 `Task` 工具派子代理**：

- **独立的探索 / 搜索 / 读代码**：派 `search` 子代理，并行多个
- **独立的实现任务**（已规划清楚、跨文件但无依赖）：派 `general_purpose_task` 子代理
- **能并行就并行**：一条消息里发多个 Task 调用
- **子代理红线**：子代理只读 / 只写文件，**禁止子代理 git commit / merge / push**（历史 P0 就是子代理并行提交时序冲突）。所有 git 操作由主 agent 串行执行
- **主 agent 职责**：拆任务 → 派子代理 → 收结果 → 主 agent 做 git 提交 → 验证回归 → 列下一批待办

## 4. 频繁提交（窗口可能随时挂）
**教训：本环境会话窗口不稳定，挂过多次。** 对策：

- 每完成一个**独立单元**（一个 pass / 一个 crate / 一组修复）立即 `git commit -s`
- 不要攒一堆才提交 —— 窗口一挂，未提交的工作全丢
- 每完成一个单元就跑一次 `cargo build && cargo test` 确认没崩，再开下一个
- 工作中途也要更新 Continuity Log（哪怕写"进行中"），别等最后才写遗言

## 5. git 安全（防 P0，血泪教训）
- 用**具体路径** `git add <file>`，禁止 `git add -A` / `git add .`（避免误加敏感文件）
- commit 用 `git commit -s`（DCO sign-off）
- **子代理只读 / 只写文件，禁止子代理 git commit / merge / push**，所有 git 操作由主 agent 串行执行
- 不碰敏感文件（.env / credentials 等）
- 不做破坏性操作（force push / reset --hard / clean -f），除非用户明确要求

## 6. 完成 + 合并 + 写遗言
按"没崩就合并"策略（**到点后**执行）：

1. 跑全量回归（第 1 步四件套）确认全绿
2. 在 `AGENTS.md` 末尾 Continuity Log **追加一段遗言**：
   - 当前状态
   - 已完成（按时序列出 commit）
   - 回归验证结果（各项计数）
   - 设计哲学遵守说明
   - 下一步（优先级排序）
3. 合并回 `main`：
   ```bash
   git checkout main && git merge --no-ff feat/neutron-<8hex>
   ```
4. 遗言是给下一任的接力棒，**必须写清下一步优先级**，否则下任要重新猜

## 7. 避嫌规范
输出与新写注释避免使用某些英文技术词的中文翻译（中文语境敏感），改用"底层存储"等中性表述。现有文件名 / 类型名保持现状（重命名核心类型是大改动，需单独一轮做）。

## 8. 代码规范
- **注释**：中文注释，技术术语保留英文（packed buffer / SSA / MLIR / cost model / CSE / IEEE754 等）
- **文件头**：每个 `.rs` 顶部用 `//!` 模块注释说明职责
- **Imports**：`use` 按 std → external → crate 分组
- **错误**：统一 `thiserror`，`base::NeutronError` + `base::Result<T>`
- **unsafe**：集中在 `base/src/storage.rs`，上层只暴露 Safe API
- **License**：Apache-2.0；语言 Rust（edition 2021，MSRV 1.75）；9-crate workspace（无 `neutron-` 前缀）

## 9. 设计哲学红线（不可擅自变更）
- **不要模式匹配**：不硬编码 `MatMul+Add→Linear` 这类贪心模式
- **用简单代数规则**：`x+0=x`、`x*1=x`、`x-x=0`（保守，NaN 风险默认不启用）
- **浮点结构优化**：针对 IEEE754 位级 trick（项目招牌，重点投入；不在简单代数/常量传播上耗时间——真实 ML 算子计算里基本不出现）
- **IO 同样性**：CSE 公共子表达式消除
- **三阶段 pipeline**：拆细（一对多 decompose）→ 重排（algebra + float + CSE）→ 融合（多对一 fuse，带 cost model）
- **规则用函数实现**，后期再抽象宏
- **cost model 现在就做**：FLOPs + memory access + launch overhead，CUDA/NPU/CPU 不同系数
