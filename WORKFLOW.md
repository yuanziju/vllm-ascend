# WORKFLOW.md — 新 agent 上手标准流程

> 每次新会话开始，agent 必须按此文件执行。用户不想每次重复讲。
> 本文件与 AGENTS.md 配合：AGENTS.md 是项目宪法（设计哲学 + 防 P0 规则 + Continuity Log 遗言），本文件是操作 checklist。

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

## 3. 工作 + 频繁提交（窗口可能随时挂）
**教训：本环境会话窗口不稳定，挂过多次。** 对策：

- 每完成一个**独立单元**（一个 pass / 一个 crate / 一组修复）立即 `git commit -s`
- 不要攒一堆才提交 —— 窗口一挂，未提交的工作全丢
- 每完成一个单元就跑一次 `cargo build && cargo test` 确认没崩，再开下一个
- 工作中途也要更新 Continuity Log（哪怕写"进行中"），别等最后才写遗言

## 4. git 安全（防 P0，血泪教训）
- 用**具体路径** `git add <file>`，禁止 `git add -A` / `git add .`（避免误加敏感文件）
- commit 用 `git commit -s`（DCO sign-off）
- **子代理只读 / 只写文件，禁止子代理 git commit / merge / push**，所有 git 操作由主 agent 串行执行（历史 P0 就是子代理并行提交时序冲突导致）
- 不碰敏感文件（.env / credentials 等）
- 不做破坏性操作（force push / reset --hard / clean -f），除非用户明确要求

## 5. 完成 + 合并 + 写遗言
按"没崩就合并"策略：

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

## 6. 避嫌规范
输出与新写注释避免使用某些英文技术词的中文翻译（中文语境敏感），改用"底层存储"等中性表述。现有文件名 / 类型名保持现状（重命名核心类型是大改动，需单独一轮做）。

## 7. 代码规范
- **注释**：中文注释，技术术语保留英文（packed buffer / SSA / MLIR / cost model / CSE / IEEE754 等）
- **文件头**：每个 `.rs` 顶部用 `//!` 模块注释说明职责
- **Imports**：`use` 按 std → external → crate 分组
- **错误**：统一 `thiserror`，`base::NeutronError` + `base::Result<T>`
- **unsafe**：集中在 `base/src/storage.rs`，上层只暴露 Safe API
- **License**：Apache-2.0；语言 Rust（edition 2021，MSRV 1.75）；9-crate workspace（无 `neutron-` 前缀）

## 8. 设计哲学红线（不可擅自变更）
- **不要模式匹配**：不硬编码 `MatMul+Add→Linear` 这类贪心模式
- **用简单代数规则**：`x+0=x`、`x*1=x`、`x-x=0`（保守，NaN 风险默认不启用）
- **浮点结构优化**：针对 IEEE754 位级 trick
- **IO 同样性**：CSE 公共子表达式消除
- **三阶段 pipeline**：拆细（一对多 decompose）→ 重排（algebra + float + CSE）→ 融合（多对一 fuse，带 cost model）
- **规则用函数实现**，后期再抽象宏
- **cost model 现在就做**：FLOPs + memory access + launch overhead，CUDA/NPU/CPU 不同系数
