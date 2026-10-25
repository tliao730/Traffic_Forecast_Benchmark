# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **traffic forecasting benchmark** that compares model families across the LargeST dataset (CA/GLA/GBA/SD subsets, years 2018/2019, 15-minute intervals). The goal is a like-for-like comparison of GNNs, zero-shot Foundation Models, SSMs, and simple baselines under one shared evaluation pipeline. In-training SSM compression (CompreSSM) is one feature of the SSM family, not the benchmark's sole focus.

The models compared:
- **GNNs**: AGCRN, ASTGCN, D2STGNN, DCRNN, DGCRN, DSTAGNN, GWNet, STGCN, STGODE, STTN, HL
- **Foundation Models (zero-shot)**: Moirai-2 (`moirai2.py`), Moirai-Small (`moirai_small.py`, moirai-1.0-R-small), Chronos-Bolt (`chronos_bolt.py`), Sundial, Kairos, TimesFM, FlowState, Toto, TabPFN-TS
- **SSMs (trained)**: Mamba-v2, LRU (CompreSSM compression), S4, LinearSSM (base)
- **Baselines**: DLinear, NLinear, ML ensemble, feedforward, naive

**CompreSSM** = in-training compression of SSMs via balanced truncation / Hankel Singular Values, currently implemented for LRU only.

## Environment & Running

All commands are run from `benchmark/` using `uv`. Each model family has its own `pyproject.toml` / virtual environment.

```bash
# FM models — all use the moirai project env except flowstate/kairos/sundial/tabpfn_ts/timesfm/toto
cd benchmark
uv run --project fm/moirai python -m fm.moirai.moirai2
# SSM training reuses the moirai env but runs the ssm/ scripts by path:
uv run --project fm/moirai python ssm/mamba.py       # Mamba-v2
uv run --project fm/moirai python ssm/linear_ssm.py  # LinearSSM base
uv run --project fm/moirai python ssm/lru.py         # LRU
uv run --project fm/moirai python ssm/s4.py          # S4

uv run --project fm/timesfm python -m fm.timesfm.timesfm
uv run --project fm/sundial  python -m fm.sundial.sundial
uv run --project fm/kairos   python -m fm.kairos.kairos
uv run --project fm/flowstate python -m fm.flowstate.flowstate

# GNN models
uv run --project gnn python -m gnn.agcrn

# ML baselines
uv run --project ml python -m ml.ml_ensemble
```

FM entrypoints **must be run as modules** (`python -m fm.xxx.xxx`) from `benchmark/`, never as scripts directly.

## Config System

The benchmark is config-driven. `BENCHMARK_CONFIG` env var points to a YAML file that overrides `benchmark/configs/default.yaml`.

```bash
# Run FM on SD 2019 test set
BENCHMARK_CONFIG=configs/fm_sd_2019.yaml uv run --project fm/moirai python -m fm.moirai.moirai2

# Run SSM training on SD 2018
BENCHMARK_CONFIG=configs/mamba_train.yaml uv run --project fm/moirai python ssm/lru.py --year 2018
```

Key config fields (`BenchmarkConfig` in `benchmark/config.py`):
- `short_datasets` / `med_long_datasets`: space-separated gift_eval dataset names (e.g. `sd/2019/15T`)
- `gift_eval_datasets_path`: path to `dataset/LargeST/gift_eval`
- `result_root`: where metrics CSVs are saved (usually `/scratch/bcqc/tliao2/TrafficFM/results`)
- `hf_home`: HuggingFace cache (set to `/u/tliao2/.cache/huggingface/hub` on Delta)
- `terms`: which gift_eval terms to run; the benchmark configs set `["long"]` (see per-step metrics below)
- `profile_windows`: 0 = the benchmark's gift_eval-capped 20 windows. A positive N is a
  *profiling* run (hour-of-day figures): results go to `result_root/<model>_prof{N}/`,
  never into the benchmark tables. `config.result_model_name()` is the one place that
  suffix is built. The GNN side takes the same value as `--profile_windows`.

Training-specific configs (SSM/Mamba) just set `result_root`, `gift_eval_datasets_path`, and `device`; dataset/year are passed as CLI args.

## SLURM Job Submission (Delta HPC)

Jobs are submitted via scripts under `scripts/run/`, organized by family:
`scripts/run/{gnn,fm,ssm,linear}/{model}/run_{model}_{region}_{year}.sh`.
Account: `bdem-delta-gpu`, partition: `gpuA100x4`.

```bash
sbatch scripts/run/ssm/lru/run_lru_sd_2018.sh
sbatch scripts/run/ssm/s4/run_s4_sd_2019.sh
sbatch scripts/run/ssm/mamba_v2/run_mamba_v2_sd_2019.sh
sbatch scripts/run/fm/timesfm/run_timesfm_sd_2019.sh
```

Training run scripts (GNN + SSM) support **segmented training**: they queue a
follow-up job with `--dependency=afterany` before training starts, so a job
killed by the 48h timeout auto-resumes from its last checkpoint; on clean
finish a done-marker is written and the follow-up is cancelled. A crash (e.g.
OOM, non-timeout) stops the chain instead of retrying.

### Job Memory by Region

Host RAM scales roughly linearly with sensor count (SD=716, GBA=2352, GLA=3834,
CA=8600), so a single `--mem` value across regions does not work. Measured peak
RSS for the trained families is ~4–5G at SD, ~5–12G at GBA, ~14–20G at GLA, and
above 32G at CA — the old flat `--mem=32G` was the direct cause of OOM kills on
`ssm/lru/ca`, `gnn/agcrn/ca` and `gnn/stgcn/ca`, and of `gnn/gwnet/ca` running
at 99% of its limit. Current requests:

| family | sd | gba | gla | ca |
|---|---|---|---|---|
| ssm / gnn | 32G | 32G | 48G | 64G |
| linear | 32G | 32G | 32G | 64G |
| fm | 64G | 64G | 64G | 64G |
| ml / simple / ensemble (cpu) | 4G | 8G | 12G | 24G |

64G at CA is deliberate: it is exactly the per-GPU share of a 256G / 4-GPU
A40x4 or A100x4 node, so asking for more lengthens queue time. Use the same
table when adding scripts for a new model. Two deliberate outliers:
`gnn/stgcn/ca/2018` keeps `128G` (the only CA run verified to complete, peak
31.8G) and `ssm/mamba_v2/gba/2019` keeps `128G` though its peak was 8.4G — that
bump was a misdiagnosed TIMEOUT, not an OOM.

Peak RSS of past jobs is the way to check this — `sacct -j <jobid> -o
JobID,State,ReqMem,MaxRSS,Elapsed` (MaxRSS lives on the `.batch` step).
Note that a MaxRSS equal to ReqMem means the job was capped, not that it needed
exactly that much. `fm` at 64G is over-provisioned (measured peak 0.8–4.8G).

### Log Layout

SLURM logs are nested by family / model / region / year, one file per job id:

```
log/{family}/{model}/{region}/{year}/{jobid}.out
log/{family}/{model}/{region}/{year}/{jobid}.err

log/ssm/lru/sd/2018/19768313.out
log/gnn/agcrn/gla/2019/19814835.err
log/ensemble/mean/ca/2018/20157238.out
```

The convention lives in one place, `scripts/lib/log_layout.sh`
(`log_dir_for_script`, `all_run_scripts`); every run script's
`#SBATCH --output/--error` is derived from it and is always absolute.
`log/{family}/_misc/` holds pre-migration logs that don't fit the pattern.

**SLURM does not create missing log directories** — a job whose leaf dir is
absent dies instantly with a file-open error. After adding a run script (new
model, region or year), create its dir before submitting:

```bash
bash scripts/maintenance/sync_log_dirs.sh          # idempotent; --dry-run to preview
```

Finding logs (filters are substrings of the path, any order):

```bash
bash scripts/maintenance/find_log.sh                   # 20 newest logs overall
bash scripts/maintenance/find_log.sh lru sd 2018       # newest for that cell
bash scripts/maintenance/find_log.sh -t lru sd 2018    # tail -f the newest .out
bash scripts/maintenance/find_log.sh -e gnn agcrn      # tail -f the newest .err
bash scripts/maintenance/find_log.sh -n 50 gnn         # 50 newest under log/gnn/
```

`scripts/maintenance/migrate_log_layout.sh` performed the one-off move from the
old flat `log/{family}/{jobid}_run_{model}_{region}_{year}.out` layout. It is
idempotent and skips logs of jobs still in the queue, so re-run it to sweep up
stragglers left behind by runs that were active during a migration.

HF cache must be set explicitly for FM jobs:
```bash
HF_HOME=/u/tliao2/.cache/huggingface/hub \
HF_HUB_CACHE=/u/tliao2/.cache/huggingface/hub \
BENCHMARK_CONFIG=configs/fm_sd_2019.yaml \
uv run --project fm/timesfm python -m fm.timesfm.timesfm
```

## Architecture: How Evaluation Works

All models flow through the same gift_eval pipeline:

1. `BENCHMARK_CONFIG` YAML is loaded → `config` object in `benchmark/config.py`
2. `common.eval()` iterates over datasets (short/medium/long terms), calls `predictor_factory(dataset)` to get a predictor, runs inference, and writes `all_results.csv` to `result_root/<model_name>/`
3. Metrics: MAE, RMSE, MAPE

### Per-step metrics and cross-family comparability

Inference runs **once** per dataset and is scored twice:

- `axis=None` -> the aggregate row in `all_results.csv` (unchanged, byte-compatible
  with every earlier run)
- `axis=0` -> one row per forecast step in `per_step_results.csv`, same directory

This exists because the two evaluation paths disagreed on what a horizon column
means. `common.py`'s short/medium/long terms are *cumulative averages* over steps
1..3 / 1..6 / 1..12, while the GNN family (`engine.py`, "Horizon k" lines) reports
the *k-th step* alone. Mixing them understates the gift_eval side by ~10% at H3,
~19% at H6 and ~31% at H12 (measured on D2STGNN sd/2018), so any table built from
both is not comparing like with like.

With per-step rows, the `long` term (12 steps) yields every column a results table
needs, in the GNN/literature convention:

```
H3 = step 3   H6 = step 6   H12 = step 12   Avg = mean(steps 1..12)
```

Because one long-term run produces all four numbers, **only the long term is
needed** for the headline table. That cuts ML/simple/ensemble CPU cost to ~1/3 and
means SSM/linear need to train one model per cell instead of three.

`mean(per-step) == aggregate` holds exactly for the mean-type metrics, so the two
files are always consistent.

GNN numbers live only in logs. `scripts/analysis/gnn_logs_to_per_step.py` extracts
them into the same schema (plus `window_mode` / `source_log` provenance columns) so
one table can hold both families:

```bash
python scripts/analysis/gnn_logs_to_per_step.py --out results/gnn_per_step_results.csv
```

`window_mode` is `stride_fallback` for any run made before gift_eval was installed
into the gnn env -- those rows are on different test windows than the rest of the
benchmark and must be regenerated with `--mode test` before use.

### Prediction dumps

Every entrypoint in every family can dump per-timestamp forecasts for
`scripts/analysis/plot_hourly_mae.py` / `plot_node_error_diff.py`:

| family | flag | written by |
|---|---|---|
| fm / ml / simple | `--save-predictions [--pred-out-dir D]` | `common.save_predictions_csv` |
| ssm / linear | `--save_predictions [--pred_out_dir D]` | `common.save_predictions_csv` |
| gnn / hl | `--save_predictions [--pred_out_dir D]` (with `--mode test`) | `BaseEngine.predict_and_save` via `gnn/src/utils/save_preds.py` |

Both writers emit one schema, `ds_config, sample_idx, item_id, timestamp, dim, stat, value`
(`stat` = `truth` | `mean`, where `mean` holds the median / point forecast), gzipped
(`.csv.gz`). The gift_eval-side dump reuses the forecasts `eval()` already scored, so it
matches `all_results.csv` exactly even for sampling models. The default dump dir is
`result_root/<model>[_prof{N}]/predictions/`; large dumps should go to
`--pred-out-dir /work/nvme/...` because /scratch is near quota. GNN dumps key sensors by
node index, the others by Caltrans ID; the row order of `{region}_meta.csv` joins them.

The shared interface is in `benchmark/fm/fm_utils.py` (used by every non-GNN family, not only FM):
- `build_basic_parser(description)` — `--eval-time` plus the dump flags
- `add_prediction_dump_args(parser, dashed=...)` / `resolve_prediction_dir(args, model_name)` — dump flags for the underscore-style ssm/linear CLIs
- `run_benchmark(eval_time_only, model_name, model_path, predictor_factory, batch_size, save_predictions_dir)` — dispatches to `eval()` or `eval_time()`
- `to_sample_forecasts(outputs, inputs)` — wraps numpy arrays into GluonTS `SampleForecast`
- `to_quantile_forecasts(...)` — for models that output quantiles (Moirai2, Sundial)
- `get_entry_target(entry)` — extracts target array from a dataset entry

Every predictor class must implement `.predict(test_data_input, batch_size)` returning a list of `Forecast` objects.

## SSM Models Architecture (`benchmark/ssm/`)

Each model is a runnable entrypoint (`{model}.py`) paired with its own model-definition module (`{model}_model.py`):

| Entrypoint | Model module | Model | Key feature |
|------------|--------------|-------|-------------|
| `mamba.py` | `mamba_model_v2.py` | Mamba-v2 | Selective SSM, parallel scan (no compression) |
| `linear_ssm.py` | `linear_ssm_model.py` | LinearSSM | Minimal diagonal LTI SSM, base model |
| `lru.py` | `lru_model.py` | LRU | Complex eigenvalues, gamma-normalised B, output gate |
| `s4.py` | `s4_model.py` | S4D | HiPPO-inspired init, learnable dt, FFN sublayer |

There is no shared `ssm_models.py` — each model owns its `{model}_model.py`. `resume_utils.py` is the one shared module (see checkpoint/resume below).

All SSM models share the same interface:
- Input: `(B, context_length, 1)`, Output: `(B, prediction_length)`
- CLI args: `--year`, `--dataset`, `--d_model`, `--d_state`, `--num_layers`, `--context_length`, `--windows_per_sensor`, `--num_workers`, `--log_dir`, `--force_retrain`, `--save_predictions`, `--pred_out_dir`; LRU also takes `--compress_epochs`, `--compress_energy`
- Training: 3 separate models (short/medium/long prediction horizons: 3/6/12 steps × 15min)
- Data: trains on `sd_train/{year}/15T`, validates on `sd_val/{year}/15T`, tests on `sd/{year}/15T`

The parallel scan uses O(log L) tree reduction — not sequential recurrence — in `mamba_model_v2.py` (`parallel_scan`) and the per-model LTI modules.

**CompreSSM** (in-training compression via balanced truncation / Hankel Singular Values) is implemented in `lru_model.py → compress()/compress_all(energy_threshold)` (LTI, so no calibration batch is needed).

It is **not** applied to Mamba-v2, S4 or LinearSSM. The Mamba path was removed once it was clear nothing enabled it: the run scripts passed a bare `--compress_epochs`, which with `nargs="*"` yields an empty list and so *disabled* the `[3, 6, 9, 12]` default. Every mamba run since then records zero compression events against 30-36 for LRU; the only run that did compress was SD/2018 in June 2026, whose twelve `.pre_compress_epoch{N}` checkpoints under `experiments/mamba_fm/SD/2018/` are now orphaned. Every mamba result in the results tree (GBA/GLA/CA) is uncompressed. The LTV variant of the implementation is recoverable from git history (parent of the removal commit).

LRU result directories carry a `_comp{energy}` tag; `model_name` selects `result_root/<model_name>/` and `check_done_datasets` treats rows already there as done, so an untagged compressed run would silently collide with an uncompressed one. The wandb run name stays the short model/region/year form.

LRU's Gramian solve inverts `(I4 - A⊗A)`, whose eigenvalues are
`{|λ|², |λ|², |λ|²e^{±2iθ}}` — exactly singular once `|λ|` reaches 1. Training
does drive `nu_log` there: `lam_abs = exp(-exp(nu_log))` rounds to 1.0 in
float32 once `exp(nu_log) < ~6e-8`, which killed the CA/2018 run at its epoch-9
compression. `LYAP_MARGIN = 1e-4` in `lru_model.py` caps `|λ|` inside
`_real_state_space_matrices()` so `1 - |λ|² ≥ 1e-4`; this affects the balanced-
truncation analysis only, `forward()` goes through `_discretised_params()` and
is untouched. `lru.py` additionally catches `torch.linalg.LinAlgError` around
`block.compress()` and skips that block's compression rather than losing a
multi-hour run.

Near-marginal `|λ|` inflates `Q ~ 1/(1-|λ|²)` and hence the leading Hankel
singular value, so the energy threshold is met by fewer states and truncation
becomes more aggressive than intended. Every existing LRU checkpoint has states
above the cap (GBA and SD/2019-short reach `|λ| = 1.0` exactly), so pre-fix
compression decisions are biased. The `.pre_compress_epoch{N}` checkpoints are
kept for every region/term/epoch, so those decisions can be replayed offline
without retraining.

**Checkpoint/resume** (`resume_utils.py`, shared by all four): each term saves `last_checkpoint_{term}_s{seed}.pt` at the end of every epoch (full training state: epoch, best_val, wait, model, optimizer, RNG). `peek_resume()` restores mid-training after a timeout; if `finished=True` the term is skipped. The `best_model_{term}_s{seed}.pt` files keep their old meaning (best-val weights, loaded for eval). LRU rebuilds compressed layers to match checkpoint shape on resume, since compression may have shrunk `d_state` mid-training.

Best-model checkpoints are saved under `benchmark/experiments/` (path set per-script via `--log_dir`, e.g. `experiments/ssm_bench/lru/SD/{year}/` for LRU, `experiments/mamba_fm/SD/{year}/` for Mamba) as `best_model_{term}_s{seed}.pt`.

## GNN Models Architecture (`benchmark/gnn/`)

Each model is a runnable entrypoint (`benchmark/gnn/{model}.py`, e.g. `agcrn.py`) that builds a model from `src/models/{model}.py` and trains via a per-model engine (`src/engines/{model}_engine.py`) subclassing `src/base/engine.py → BaseEngine`. CLI args are in `src/utils/args.py`.

**Per-year experiment dirs**: every entrypoint writes to `experiments/{MODEL}/{DATASET}/{YEAR}/` so 2018/2019 runs don't overwrite each other. Two files per run: `last_checkpoint_s{seed}.pt` (resume state) and `final_model_s{seed}.pt` (best weights, loaded for eval).

**Checkpoint/resume** (`BaseEngine.save_checkpoint()`/`load_checkpoint()`): saved every epoch; `train()` resumes from `last_checkpoint_s{seed}.pt`, and if its `finished` flag is set, skips training and runs test only (**delete the checkpoint to retrain from scratch** — there is no `--force_retrain` for GNNs). D2STGNN/DGCRN engines override `_extra_checkpoint_state()` to also persist curriculum-learning state (`cl_len`/`task_level`).

**Large-N memory (GBA/GLA/CA)**: the O(N²) graph ops OOM as node count grows (SD=716, GBA=2352, GLA=3834, CA=8600). Each model exposes an opt-in CLI knob that trades compute for peak memory via chunking + `torch.utils.checkpoint`:
- AGCRN: `--checkpoint_steps` (checkpoint each RNN timestep)
- DGCRN: `--checkpoint_steps` (+ `--no_checkpoint_encoder`/`--no_checkpoint_decoder` to checkpoint only one side)
- DSTAGNN: `--checkpoint_cheb_conv` (checkpoint each (time_step, k) cheb-conv pair)
- STTN: `--spatial_attn_chunk_size N` (chunk the spatial-attention batch dim)
- D2STGNN: `--gconv_chunk_size N` (chunk the graph-conv matmul batch dim)
- DCRNN: `--lr_milestones` (configurable LR decay epochs; CA diverges before the default `[10,50,90]`)

## Dataset Structure

```
dataset/LargeST/gift_eval/
  sd_train/2018/15T/   # training split
  sd_val/2018/15T/     # validation split
  sd/2018/15T/         # test split (used by gift_eval eval pipeline)
  sd_train/2019/15T/
  ...
```

Dataset names follow the pattern `{region}[_train|_val]/{year}/{freq}`. Regions: `sd`, `ca`, `gba`, `gla`.

## Results Location

- FM zero-shot results: `/scratch/bcqc/tliao2/TrafficFM/results/{model_name}/all_results.csv`
- Scratch also at: `/u/tliao2/TrafficFM/results_scratch/` (some older runs copied here)
- SSM experiment checkpoints: `/u/tliao2/TrafficFM/benchmark/experiments/`

## External Repo Dependencies (FM models)

Some FM models require cloning external repos into `benchmark/fm/envs/`:

```bash
# FlowState
git clone https://github.com/ibm-granite/granite-tsfm.git benchmark/fm/envs/flowstate/granite-tsfm
# Kairos
git clone https://github.com/foundation-model-research/Kairos.git benchmark/fm/envs/kairos/Kairos
# Toto
git clone https://github.com/DataDog/toto.git benchmark/fm/envs/toto/toto
```

`load_model.py → setup_model_runtime()` handles path resolution; env vars (`GRANITE_TSFM_PATH`, `KAIROS_PATH`, `TOTO_PATH`) can override the default search paths.

## Adding a New SSM Model

1. Add `benchmark/ssm/{model_name}_model.py` with a model class exposing `forward((B,L,1)) → (B,pred_len)` and `param_num()`
2. Create `benchmark/ssm/{model_name}.py` following the pattern of `lru.py`: training loop (wire in `resume_utils` for checkpoint/resume) + Predictor class + `main()`; call `add_prediction_dump_args(parser, dashed=False)` and pass `save_predictions_dir=resolve_prediction_dir(args, model_name)` to `run_benchmark`
3. Add SLURM scripts in `scripts/run/ssm/{model_name}/run_{model_name}_{region}_{year}.sh` following existing templates (including the segmented-training auto-resume block), with `#SBATCH --output=/u/tliao2/TrafficFM/log/ssm/{model_name}/{region}/{year}/%j.out` (same for `--error`/`.err`)
4. Run `bash scripts/maintenance/sync_log_dirs.sh` so the new leaf log dirs exist — SLURM will not create them
5. Use `BENCHMARK_CONFIG=configs/mamba_train.yaml` for training jobs
