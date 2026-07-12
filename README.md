# Neutron

A Rust-based ML compiler that lowers high-level models (ONNX / custom DSL / PyTorch) through architecture-independent optimization, architecture-specific lowering, and instruction selection into target backends (CUDA / Ascend NPU / CPU).

## Status

11-crate Rust workspace with a working end-to-end pipeline (frontend → optimizer → arch → isel → regalloc → backend). Four backend codegen targets are stubbed/implemented: CUDA (wmma/wgmma/tcgen05), Triton (TMA), Metal (simdgroup), CANN (AscendC). v0.1.1.

## Design

- **IR**: MLIR-style (everything is an op + region), pure-functional SSA, tagged value IDs (type tag encoded in the ID), static types with shapes in the type system.
- **Storage**: Continuous packed buffer + `unsafe` + Safe wrappers. `ID = offset`, O(1) access.
- **Optimization**: Three-stage pipeline (decompose → reorder [algebra + float + CSE] → fuse). No hardcoded pattern matching — only simple algebraic rules, IEEE754 bit-level float tricks, CSE, and cost-model-driven fusion.
- **Cost model**: FLOPs + memory access + launch overhead, with per-target coefficients (CUDA / NPU / CPU).

## Workspace

| crate | responsibility |
|-------|----------------|
| `base` | IR core: `RawGraph` (packed buffer, unsafe) + `Graph` (Safe API), `NeutronError`, types, `OpKind`, `Pass`/`PassContext` |
| `common` | `Target`, `OptLevel`, `IdGen`, `Arena`, `Config`, `dump_graph` |
| `frontend` | ONNX / DSL / PyTorch parsing (currently placeholders) |
| `optimizer` | Three-stage pipeline + passes (DCE, Verify, algebra, float_opts, CSE, decompose, fuse) + cost_model |
| `arch` | `ArchGraph` + `lower()` (1:1 op → native kernel) + device descriptors |
| `lisp` | S-expr interpreter (parser + interp) for isel rules |
| `isel` | Instruction selection `select()` |
| `regalloc` | Chaitin-Briggs graph-coloring register allocator (liveness → interference → coalescing → coloring → spill) |
| `backend` | Codegen for CUDA / Triton / Metal / CANN; `SourceLang` enum + `emit_for()` dispatch |
| `interface` | Single public API `compile()` chaining frontend → optimizer → arch → isel → regalloc → backend |
| `cli` | `neutron` binary (`--target` / `--opt` / `--dump` / `-o` / `--version` / `--help`) |

## Build & Test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump
```

## CLI

```bash
# 编译 ONNX → CUDA 后端源码，输出到 stdout
neutron model.onnx --target cuda --opt 2

# 输出到文件
neutron model.onnx --target npu --opt 3 -o kernel.cpp

# 查看 IR 调试信息 (走 stderr)
neutron model.onnx --target cpu --opt 1 --dump

# 帮助 / 版本
neutron --help
neutron --version
```

退出码：`0` = 成功，`1` = 运行时错误，`2` = 用法错误（缺值、未知 flag 等，便于脚本区分）。

## License

Apache-2.0.
