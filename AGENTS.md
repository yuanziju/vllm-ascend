# AGENTS.md

Guidance for AI coding agents working in the `vllm-ascend` repository.

## Project Overview

`vllm-ascend` is a community-maintained hardware plugin that runs [vLLM](https://github.com/vllm-project/vllm) seamlessly on Huawei Ascend NPUs. It implements the hardware-pluggable interface proposed in [vLLM RFC #11162](https://github.com/vllm-project/vllm/issues/11162), decoupling Ascend NPU integration from the vLLM core.

- License: Apache 2.0
- Hardware: Atlas 800I A2 / A2 Training / 800I A3 / A3 Training / 300I Duo (experimental)
- OS: Linux
- Python: >=3.9, <3.12
- Runtime deps: CANN >= 8.2.rc1, PyTorch >= 2.7.1, torch-npu >= 2.7.1.dev20250724, vLLM (same version as vllm-ascend)

The plugin registers itself via two vLLM entry points (see [setup.py](file:///workspace/setup.py)):
- `vllm.platform_plugins` -> `ascend = vllm_ascend:register` (registers `NPUPlatform`)
- `vllm.general_plugins` -> `ascend_enhanced_model = vllm_ascend:register_model`

## Repository Layout

- `vllm_ascend/` — Main Python package (the plugin)
  - `platform.py` — `NPUPlatform` (Ascend platform registration with vLLM)
  - `ascend_config.py` — Ascend-specific configuration
  - `envs.py` — All environment variables (build + runtime). **Read this first** when an env var is referenced.
  - `attention/` — Attention backends (v1, MLA v1, attention masks)
  - `distributed/` — Tensor parallel, pyhccl, mooncake / llmdatadist KV connectors
  - `models/` — Ascend-optimized model implementations (DeepSeek V2/V3/MTP, Qwen2/Qwen2.5-VL, Qwen3, Qwen3-MoE, Pangu-MoE)
  - `ops/` — Custom ops (fused MoE, linear, layernorm, rotary embedding, vocab parallel embedding, MoE dispatcher)
  - `quantization/` — W8A8 / W4A8 (dynamic) quantization
  - `worker/` — v1 worker, model runner, MTP/EAGLE proposers, npu input batch
  - `lora/` — LoRA via punica wrapper on NPU
  - `torchair/` — TorchAir graph mode integration (deepseek, qwen models)
  - `multistream/` — Multi-stream execution
  - `device_allocator/` — CAMem device allocator
  - `patch/` — Runtime patches applied to vLLM (platform + worker)
  - `compilation/` — ACL graph
- `csrc/` — C++ kernels (AscendC) + pybind11 bindings. Built via CMake when `COMPILE_CUSTOM_KERNELS=1`.
- `cmake/` — CMake utilities
- `tests/`
  - `ut/` — Unit tests (per-module: attention, ops, distributed, models, quantization, ...)
  - `e2e/` — End-to-end tests (`singlecard/`, `multicard/`, `pd_disaggreate/`, `doctests/`, `models/`)
- `examples/` — Offline inference, DP server, disaggregated prefill, eplb
- `benchmarks/` — Op micro-benchmarks + serving/throughput/latency suites
- `docs/` — Sphinx docs (English sources in `docs/source/`, Chinese translations in `docs/source/locale/zh_CN/`)
- `tools/` — Lint/check helpers (`mypy.sh`, `check_repo.sh`, `enforce_regex_import.py`, ...)
- `.github/workflows/` — CI: build images, pre-commit, vllm_ascend_test (singlecard/multicard/310p/pd), nightly benchmarks, release

## Build & Install

Build requires CANN toolkit and torch-npu installed. Key env vars (see [envs.py](file:///workspace/vllm_ascend/envs.py)):

- `SOC_VERSION` — Ascend chip version, default `ASCEND910B1`. **Required** for `310` series (310 only supports custom kernels).
- `COMPILE_CUSTOM_KERNELS` — `1` (default) compiles C++/AscendC kernels; `0` skips (also disables sleep mode).
- `MAX_JOBS` — Parallel compile jobs.
- `CMAKE_BUILD_TYPE` — `Release` (default) / `Debug` / `RelWithDebugInfo`.
- `ASCEND_HOME_PATH` — CANN toolkit root, default `/usr/local/Ascend/ascend-toolkit/latest`.
- `VLLM_VERSION` — Set when vLLM is installed from source so its version differs from the tagged release.

Editable install (development):
```bash
pip install -v -e .
# With custom kernels for a specific SoC:
SOC_VERSION=ASCEND910B1 pip install -v -e .
```

## Testing

Tests run on actual Ascend NPU hardware (no CPU fallback for most). Unit and e2e tests use pytest.

```bash
# Unit tests
pytest tests/ut/<module>/test_<name>.py

# E2E single-card
pytest tests/e2e/singlecard/test_<name>.py

# Doctests
bash tests/e2e/run_doctests.sh
```
Dev test deps: `requirements-dev.txt` (pytest, lm-eval, ray, xgrammar, sentence_transformers, ...).

## Linting & Formatting

Run everything via pre-commit:
```bash
bash format.sh        # local: pre-commit run --all-files
bash format.sh ci     # CI: also runs manual-stage mypy (3.9/3.10/3.11/3.12)
```
Hooks (see [.pre-commit-config.yaml](file:///workspace/.pre-commit-config.yaml)):
- **yapf** — Python formatting for `vllm_ascend/` and `tests/` (excludes `.github`, `benchmarks`, `examples`, `docs`)
- **ruff** — Lint + autofix; `ruff-format` only for `benchmarks/` and `examples/`
- **isort** — Import ordering
- **codespell** + **typos** — Spell check (with allowlist: CANN, cann, ASCEND, ascend, NNAL, ...)
- **pymarkdown** — Markdown linting (config in [pyproject.toml](file:///workspace/pyproject.toml))
- **actionlint** — GitHub Actions workflows
- **mypy** — Manual stage (CI only), Python 3.9-3.12
- **enforce-import-regex** — `import regex as re` is enforced instead of stdlib `re`
- **check_python_src_init** — Enforces `__init__.py` in Python packages
- **signoff-commit** — DCO sign-off required on every commit (commit-msg stage)
- **png-lint** — Excalidraw PNG exports must be cropped
- Filenames must not contain spaces

Lint deps: `pip install -r requirements-lint.txt` then `pre-commit install`.

## Conventions

- **DCO**: Every commit must be signed off (`Signed-off-by: Name <email>`). The `signoff-commit` hook adds this automatically if `git config user.name/email` are set.
- **Imports**: Use `import regex as re` (not stdlib `re`) — enforced by hook.
- **Python packages**: Every Python source directory under `vllm_ascend/` and `tests/` must have `__init__.py`.
- **File headers**: Source files carry the Apache 2.0 license header + "This file is a part of the vllm-ascend project." line (and "Adapted from ..." when ported from vLLM).
- **Commit messages**: Follow the project's conventional style; reference issues when applicable.
- **Branch policy**: `main` tracks vLLM main; `vX.Y.Z-dev` branches track vLLM releases. Bug fixes only on dev branches; no new release tags on `v0.7.3-dev`.

## Common Agent Tasks

- **Adding a model**: See `docs/source/developer_guide/modeling/adding_a_new_model.md`. Place under `vllm_ascend/models/`, register in `register_model`.
- **Adding an op**: Custom AscendC kernels live in `csrc/kernels/`; Python bindings in `csrc/torch_binding*.cpp`; wrappers in `vllm_ascend/ops/`. Rebuild with `COMPILE_CUSTOM_KERNELS=1`.
- **Env-var changes**: Update `vllm_ascend/envs.py` (the `env_variables` dict is scraped by docs generator — keep the `begin-env-vars-definition` / `end-env-vars-definition` markers).
- **Platform/worker patches**: `vllm_ascend/patch/` applies runtime monkey-patches to vLLM; keep `patch_common/` (shared) vs `patch_main/` (vLLM-main-only) separation.

## Useful References

- Quick start: https://vllm-ascend.readthedocs.io/en/latest/quick_start.html
- Installation: https://vllm-ascend.readthedocs.io/en/latest/installation.html
- Contributing: https://vllm-ascend.readthedocs.io/en/latest/developer_guide/contribution/index.html
- Support matrix: `docs/source/user_guide/support_matrix/`
- Versioning policy: `docs/source/community/versioning_policy.md`
