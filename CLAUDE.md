# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

openpi is a robotics model library from Physical Intelligence (π). It contains JAX and PyTorch implementations of three vision-language-action (VLA) models: **π₀** (flow-based), **π₀-FAST** (autoregressive), and **π₀.₅** (upgraded π₀ with knowledge insulation). The repo provides base model checkpoints pre-trained on 10k+ hours of robot data, plus fine-tuned checkpoints for specific robots/platforms (ALOHA, DROID, LIBERO).

## Environment & package management

- Python >= 3.11, managed with [uv](https://docs.astral.sh/uv/). Dependencies in `pyproject.toml`.
- `uv sync` to install all dependencies. `GIT_LFS_SKIP_SMUDGE=1` is needed when pulling LeRobot as a dependency.
- `uv pip install -e .` for editable install after sync.
- The workspace has a sibling package at `packages/openpi-client/` (workspace member, lighter-weight client-only package).
- Pre-commit hooks: ruff lint + format (line length 120), uv-lock.

## Common commands

```bash
# Install / sync
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Lint & format
uv run ruff check .
uv run ruff format .

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest src/openpi/models/model_test.py
uv run pytest src/openpi/models/model_test.py::test_pi0_model

# Run tests excluding slow/manual markers
uv run pytest -m "not manual"

# Pre-commit
uv run pre-commit run --all-files
```

## Training

```bash
# 1. Compute normalization stats first (required before training)
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 2. JAX training (primary training path)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite

# PyTorch training (single GPU)
uv run scripts/train_pytorch.py pi05_libero --exp_name my_experiment --save_interval 20000

# PyTorch training (multi-GPU via DDP)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_libero --exp_name my_experiment

# Resume PyTorch training from checkpoint
uv run scripts/train_pytorch.py pi05_libero --exp_name my_experiment --resume
```

## Inference / policy server

```bash
# Serve a policy (checkpoint-based)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# Convert JAX model to PyTorch
uv run examples/convert_jax_model_to_pytorch.py --config_name pi05_droid --checkpoint_dir /path/to/jax/checkpoint --output_path /path/to/output
```

## Architecture

### Core layers

1. **`src/openpi/models/`** — JAX model implementations using Flax NNX.
   - `model.py`: `BaseModel` (abstract, extends `nnx.Module`) and `BaseModelConfig` (dataclass). `Observation` is a typed container for images, state, prompts, and masks. `Actions` = `float32[*b ah ad]`.
   - `pi0.py`: π₀ model — PaliGemma + action expert, flow-matching head.
   - `pi0_fast.py`: π₀-FAST model — autoregressive action tokenizer based on FAST.
   - `pi0_config.py`: `Pi0Config` — configuration for π₀/π₀.₅ (action_dim=32, action_horizon=50). `pi05=True` switches to π₀.₅ behavior (discrete state input, adaRMS norm).
   - `gemma.py` / `gemma_fast.py`: Gemma language model backbones (PaliGemma variants).
   - `siglip.py`: SigLIP vision encoder.
   - `lora.py`: LoRA adaptation support.

2. **`src/openpi/models_pytorch/`** — PyTorch model implementations.
   - `pi0_pytorch.py`: `PI0Pytorch` — PyTorch port of π₀/π₀.₅ with `torch.compile` support.
   - `gemma_pytorch.py`: PyTorch Gemma backbone.
   - `transformers_replace/`: Patches to the `transformers` library (AdaRMS, KV cache control, activation precision). Must be manually copied into the virtualenv's transformers installation.

3. **`src/openpi/training/`** — Training pipeline.
   - `config.py`: `TrainConfig` (name + model config + data config + training hyperparameters). All predefined configs live in the `_CONFIGS` list (ALOHA, DROID, LIBERO, debug configs for both π₀/π₀-FAST/π₀.₅). `get_config(name)` resolves a config by name. `DataConfig` defines data source, normalization, transforms. `ModelTransformFactory` creates model-specific transform groups.
   - `data_loader.py`: LeRobot dataset loading, `TransformedDataset`, data pipeline abstraction.
   - `checkpoints.py`: Checkpoint save/restore via orbax.
   - `weight_loaders.py`: Weight loading strategies — `CheckpointWeightLoader`, `PaliGemmaWeightLoader`, LoRA merging, regex-based parameter filtering.
   - `optimizer.py`: Optimizer configuration.
   - `sharding.py`: FSDP sharding utilities for multi-GPU training.

4. **`src/openpi/policies/`** — Policy wrappers for inference.
   - `policy.py`: `Policy` class — wraps a model with input/output transforms (normalization, tokenization, image resizing) and supports both JAX (`nnx_utils.module_jit`) and PyTorch models. `PolicyRecorder` records policy inputs/outputs to disk.
   - `policy_config.py`: `create_trained_policy()` — factory that loads a checkpoint, auto-detects JAX vs PyTorch format, and assembles the full transform chain.
   - `*_policy.py` (aloha, droid, libero): Platform-specific data mapping (robot observations → model format, model outputs → robot actions).

5. **`src/openpi/serving/`** — Model serving.
   - `websocket_policy_server.py`: WebSocket-based policy server (msgpack serialization). Client connects, sends observations, receives actions.

6. **`src/openpi/shared/`** — Shared utilities.
   - `array_typing.py`: JAX/PyTorch/NumPy array type annotations with `@typecheck`.
   - `normalize.py`: `NormStats`, `RunningStats`, z-score and quantile normalization. Stats stored as `norm_stats.json` in asset directories.
   - `download.py`: GCS checkpoint download with local caching (`~/.cache/openpi`, overridable via `OPENPI_DATA_HOME`).
   - `image_tools.py`: Image resize/pad utilities.
   - `nnx_utils.py`: JAX NNX utilities (`module_jit` for JIT-compiling NNX module methods).

7. **`src/openpi/transforms.py`** — Data transformation pipeline.
   - `DataTransformFn` protocol: `(DataDict) -> DataDict`. Transform functions are composed and applied in sequence.
   - `Group`: pairs of input and output transforms.
   - Built-in transforms: `Normalize`/`Unnormalize`, `TokenizePrompt`, `ResizeImages`, `InjectDefaultPrompt`, `PadStatesAndActions`, etc.

8. **`packages/openpi-client/`** — Lightweight client package (no JAX/PyTorch dependency).
   - `base_policy.py`: Abstract `BasePolicy` interface.
   - `websocket_client_policy.py`: Client that connects to `WebsocketPolicyServer`.
   - `msgpack_numpy.py`: MsgPack serialization with NumPy array support.
   - `runtime/`: Robot-facing runtime utilities.

9. **`scripts/`** — Entry points.
   - `train.py`: JAX training loop (the primary training path).
   - `train_pytorch.py`: PyTorch training loop with DDP support.
   - `serve_policy.py`: Policy server launcher (tyro CLI).
   - `compute_norm_stats.py`: Precompute normalization statistics for a dataset.

### Data flow

**Training:** Raw LeRobot dataset → `DataConfig` transforms (repack → data transforms → normalize → model transforms) → `Observation` + `Actions` → `BaseModel.compute_loss()` → optimizer step. Checkpoints saved to `checkpoints/<config_name>/<exp_name>/<step>/`.

**Inference:** Raw robot observations → input transforms (resize → tokenize prompt → normalize) → `Observation.from_dict()` → `BaseModel.sample_actions()` → output transforms (unnormalize → platform-specific mapping) → action chunk dict.

### Key design points

- **Model == config + weights:** `BaseModelConfig.create(rng)` initializes a fresh model. `BaseModelConfig.load(params)` / `load_pytorch(train_config, path)` restores from weights. The `TrainConfig` bundles model config + data config + training hyperparams.
- **Transform chain:** All data processing is a composable pipeline of `DataTransformFn` callables. `Group` tracks separate input and output chains. Apply order: repack → data transforms → normalize → model transforms (and reverse for outputs).
- **JAX/PyTorch dual support:** `Policy` and `create_trained_policy` auto-detect format by checking for `model.safetensors`. JAX models use `nnx_utils.module_jit`; PyTorch models use `torch.compile` and explicit device management.
- **Assets system:** Normalization stats are stored per "asset id" (e.g., `trossen`, `droid`, `libero`) inside `assets/` directories. Configs reference assets via `AssetsConfig(asset_id=...)`. Training copies assets into the checkpoint.
- **No multi-node JAX training yet** (only single-node FSDP). PyTorch supports multi-node via `torchrun`.
- **All paths use `epath.Path`** (from `etils`) for GCS/local transparent path handling.
