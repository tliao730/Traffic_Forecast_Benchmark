# Reproducing the TrafficFM benchmark

This is the checklist for rebuilding the paper's results after the original
working copy on NCSA Delta was deleted (October 2026). `QUICKSTART.md` is the
short version. This file adds what QUICKSTART leaves out: the pinned model
versions, the hidden data steps, and where the trained weights now live.
`CLAUDE.md` has the detailed engineering notes, such as memory sizing, the
per-step metric convention and the LRU numerics.

## 0. What lives where

| what | where | notes |
|---|---|---|
| code, configs, SLURM scripts | this repo (`benchmark/`, `scripts/run/`) | `scripts/run/` holds the exact command line of every run |
| paper sources and the results grid | `paper/` | **`paper/results_grid.csv` is the numbers of record** |
| raw metric CSVs that survived | `results/raw/<model>/` | incomplete, see below |
| GNN per-horizon metrics | `results/gnn_per_step_results.csv` | extracted from SLURM logs by `scripts/analysis/gnn_logs_to_per_step.py` |
| trained weights (GNN, SSM, linear) | Hugging Face model repo `angus3938/TrafficFM-checkpoints` (private) | listed in `results/checkpoints_manifest.csv` with sha256 |
| SLURM logs | `logs/trafficfm_log.tar.gz` in the same HF repo | the only source of the GNN "Horizon k" numbers |
| datasets | rebuilt from Kaggle (section 2) | none of it is original data |
| FM weights | Hugging Face, public | pin the revisions in section 3 |

`results/raw/` is not a complete record. A /scratch purge in June 2026 removed
the benchmark-window results of Moirai-2, Chronos-Bolt, Kairos and Sundial
(only their `_prof500` profiling runs survive), and GNN numbers were never
written to CSV. If `results/raw/` and `paper/results_grid.csv` disagree, the
grid wins.

Suggested first step after cloning: check out the tag the paper was submitted
from (`git checkout tmlr-submission`), not the branch head.

## 1. Environments

Every family has its own `uv` project and a committed `uv.lock`, so `uv sync`
reproduces the exact package set. All of them use Python 3.10.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd benchmark
for p in fm/moirai fm/timesfm fm/sundial fm/kairos fm/flowstate gnn ml feedforward; do
    uv sync --project $p
done
# fm/moirai also runs the SSM (ssm/*.py) and linear (linear/*.py) models.
# fm/tabpfn_ts and fm/toto are not in the paper; see QUICKSTART.md if needed.
```

External repos, pinned to the commits the paper used:

```bash
git clone https://github.com/foundation-model-research/Kairos.git benchmark/fm/Kairos
git -C benchmark/fm/Kairos checkout d926ddb          # found via load_model.py, or set KAIROS_PATH

git clone https://github.com/ibm-granite/granite-tsfm.git benchmark/fm/flowstate/granite-tsfm
git -C benchmark/fm/flowstate/granite-tsfm checkout f87e8bfc
export GRANITE_TSFM_PATH=$PWD/benchmark/fm/flowstate/granite-tsfm   # the FlowState run scripts set this
```

## 2. Data

Everything under `data/` and `dataset/` is derived from the LargeST release on
Kaggle. Run from the repo root, and give each step both years (2018 and 2019)
and all four regions (`sd gba gla ca`):

```bash
./scripts/data_processing/LargeST/download_dataset.sh                 # ~35 GB raw, needs ~/.kaggle/kaggle.json
for y in 2018 2019; do
    uv run ./scripts/data_processing/LargeST/process_dataset.py --year $y   # writes all four regions
    for r in sd gba gla ca; do
        uv run ./scripts/data_processing/LargeST/generate_dataset_for_training.py --dataset $r --years $y   # GNN
        uv run ./scripts/data_processing/LargeST/generate_data_for_gift_eval.py   --dataset $r --years $y   # everyone
    done
done
```

Two steps QUICKSTART does not mention. Without them every GNN fails at start-up:

```bash
# 1. The road-network adjacency has to sit next to his.npz. One file per region, shared by both years.
for r in sd gba gla ca; do
    cp data/LargeST/$r/${r}_rn_adj.npy dataset/LargeST/baseline/$r/
done

# 2. The GNN entrypoints read ./data/<region> relative to benchmark/gnn.
mkdir -p benchmark/gnn/data
for r in sd gba gla ca; do
    ln -s $PWD/dataset/LargeST/baseline/$r benchmark/gnn/data/$r
done
```

On Delta, `data` and `dataset` were symlinks into `/scratch/bcqc/tliao2/TrafficFM/`.
The tracked `dataset` symlink still points there, so re-point it (or replace it
with a real directory) before step 2.

GNNs need both `baseline/` (training, `his.npz`) and `gift_eval/` (test-window
anchors). Every other family reads only `gift_eval/`.

## 3. Foundation-model weights

The loaders call `from_pretrained(<repo id>)` without a `revision`
(`load_pretrained_with_cache` in `benchmark/fm/fm_utils.py`), so they fetch
whatever the default branch holds today. These are the snapshots the paper ran:

| model | Hugging Face repo | revision |
|---|---|---|
| Moirai-2 | `Salesforce/moirai-2.0-R-small` | `30f43ff08c8494f4943ae1521e9d4e94a0fbb389` |
| Moirai-Small | `Salesforce/moirai-1.0-R-small` | `f5bd5d01f0d67de856107d43abdd637380aae0a3` |
| Chronos-Bolt | `amazon/chronos-bolt-base` | `5d9f166d69f47aef3401367a7b842e78fe97b121` |
| TimesFM | `google/timesfm-2.0-500m-pytorch` | `dc2443792ce5516872b89b37cf1bc058c3bf0c10` |
| Sundial | `thuml/sundial-base-128m` | `3212e42564493f520593e5414af4367fc4b49226` |
| Kairos | `mldi-lab/Kairos_50m` | `b788434dd95f9893be002e6858bfb2afb0e3f360` |
| FlowState | `ibm-research/FlowState` | `ec815e416006b6a74130e1c79c6c67004c12c0c8` |

If a repo has moved on, pass `revision=<sha>` at that `from_pretrained` call.
(`google/timesfm-2.0-500m-jax` was also cached but is unused. TimesFM loads
the PyTorch checkpoint.)

## 4. Trained weights

```bash
cd benchmark
hf auth login                       # the repo is private; log in as angus3938 (or make it public)
hf download angus3938/TrafficFM-checkpoints --local-dir . --include "gnn/*" "experiments/*"
# verify
python - <<'EOF'
import csv, hashlib
for r in csv.DictReader(open("../results/checkpoints_manifest.csv")):
    h = hashlib.sha256(open(r["path_relative_to_benchmark"], "rb").read()).hexdigest()
    print("ok " if h == r["sha256"] else "BAD", r["path_relative_to_benchmark"])
EOF
```

Paths in the HF repo are relative to `benchmark/`, so the files land where the
code expects them: `gnn/experiments/<MODEL>/<REGION>/<YEAR>/final_model_s2023.pt`
and `experiments/<group>/<model>/<REGION>/<YEAR>/best_model_<term>_s2023.pt`.

### Which paper numbers these weights reproduce

`results/gnn_provenance.csv` traces every GNN/HL cell of
`paper/results_grid.csv` back to the SLURM log whose test block it was copied
from, and checks whether `final_model` was rewritten after that evaluation.
The grid's two decimals were rounded or truncated inconsistently, so matches
allow ±0.015. Of the 69 cells:

- **36 are traceable.** Re-evaluating the uploaded weights with
  `scripts/run/gnn_eval/...` should give the grid's numbers. This includes
  every CA/2019 cell, and HL, which has no weights.
- **5 CA/2018 cells (AGCRN, ASTGCN, DCRNN, DSTAGNN, STTN) cannot be
  reproduced from the uploaded weights.** The grid holds a 2026-09-06
  evaluation of the best-so-far `final_model` while training was still
  running. Training then continued and overwrote `final_model`, so re-evaluating
  now gives different (later-epoch) numbers. AGCRN's grid H12 (32.28) also
  differs from that log (32.38), probably a transcription slip.
- **19 of the 20 SD cells have no surviving source log**; only HL SD/2018 does. Their
  pattern (Avg ≈ H6) is the pre-alignment evaluation convention, and
  re-evaluating the SD/2018 weights does not reproduce them (STGCN H12: grid
  21.51, re-evaluated 28.21). Treat the SD GNN rows as carried over from an
  earlier evaluation, not as outputs of this pipeline.
- **9 more cells have no surviving log:** GBA/2019 AGCRN, DCRNN, DSTAGNN,
  GWNET, STGCN; GLA/2019 GWNET; GBA/2018 ASTGCN; GLA/2018 STGCN; and HL
  CA/2018. Most of them also have no weights.

### Status of the weights themselves (`status` in the manifest)

- **ASTGCN and DSTAGNN CA/2018 never reached `--max_epochs 40`.** Both hit
  the 48 h wall clock and the follow-up job never ran: ASTGCN stopped at epoch
  24 and DSTAGNN at 27. Their `last_checkpoint_s2023.pt` (optimizer + RNG
  state) is in the HF repo, so `--mode train` with the run script resumes to
  epoch 40.
- **AGCRN SD/2018 finished, but its weights were lost** before the upload.
- The GNN models on CA were trained for `--max_epochs 40` (AGCRN, ASTGCN,
  DSTAGNN, STTN) or 100 (the rest). The budget is in each run script.
- 25 SSM files from early SD/GBA runs predate done markers, so they are marked
  `unknown`. Their logs are in the log archive.
- Only `final_model` / `best_model` (best validation) and the two resume files
  above were kept. The other `last_checkpoint` files (18 GB) were not uploaded.

## 5. Re-running

The run scripts are written for Delta. Before using them elsewhere, replace:

- `/u/tliao2/TrafficFM`, the repo path, in about 320 files
- `/scratch/bcqc/tliao2/TrafficFM/results`, the `result_root` in `benchmark/configs/*.yaml`
- `--account=bdem-delta-gpu|bcqc-delta-gpu|bcqc-delta-cpu` and the partition names (`gpuA100x4`, `gpuH200x8`, `cpu`)

```bash
grep -rl /u/tliao2/TrafficFM scripts benchmark/configs | xargs sed -i "s|/u/tliao2/TrafficFM|$PWD|g"
bash scripts/maintenance/sync_log_dirs.sh      # SLURM will not create log directories itself
```

- **Evaluate GNNs from the downloaded weights (CPU is enough):**
  `scripts/run/gnn_eval/<model>/run_<model>_<region>_<year>.sh` (`--mode test`).
  `bash scripts/analysis/verify_gnn_alignment.sh` checks that GNN test windows
  match gift_eval's.
- **SSM / linear:** rerun the training script. A finished `best_model_*` makes
  it skip training and only evaluate (`--force_retrain` overrides).
- **FM / ML / simple / ensemble:** no training. Run `scripts/run/{fm,ml,simple,ensemble}/...`.
- **Tables:** `scripts/analysis/gnn_logs_to_per_step.py` (GNN logs to CSV),
  then `scripts/analysis/make_appendix_tables.py` (grid to LaTeX).

Every benchmark config sets `terms: ["long"]`. H3/H6/H12 are steps 3/6/12 of
the 12-step forecast and Avg is the mean over steps 1-12, read from
`per_step_results.csv`. CLAUDE.md, "Per-step metrics", explains why mixing this
with gift_eval's cumulative short/medium terms understates errors by 10-31%.
