# Neutron

A Rust-based ML compiler that lowers high-level models (ONNX / custom DSL / PyTorch) through architecture-independent optimization, architecture-specific lowering, and instruction selection into target backends (CUDA / Ascend NPU / CPU).

## Status

Early stage — 9-crate workspace scaffolded with a working end-to-end pipeline (frontend → optimizer → arch → isel). Optimization passes are being filled in incrementally.

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
| `interface` | Single public API `compile()` chaining frontend → optimizer → arch → isel |
| `cli` | `neutron` binary (`--target` / `--opt` / `--dump`) |

## Build & Test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run -p cli -- /tmp/empty.onnx --target cuda --opt 2 --dump
```

## License

Apache-2.0.
