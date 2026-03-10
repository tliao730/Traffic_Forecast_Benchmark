# Experiments eval_time — 与 benchmark 同协议预估运行时间

用与 `benchmark/run_eval_time_all.py` **相同的 eval_time 协议**（相同数据集列表、相同 estimation_samples、相同输出 JSON 结构）对 `experiments/` 下所有模型做运行时间预估，便于与 benchmark 的 foundation 模型对比。

## 环境（用项目根目录的 uv 环境即可）

官方 repo 没有单独写「experiments 用什么环境」，这里**直接复用项目根目录的 uv 环境**即可。根目录的 `pyproject.toml` / `uv.lock` 里已经带了 `torch`、`gluonts`、`gift_eval`、`numpy`、`pandas` 等，满足 experiments 和 eval_time 的需求。

### 步骤 1：在 TrafficFM 根目录同步环境

```bash
cd TrafficFM
uv sync
```

### 步骤 2（可选）：若要跑 stgode，安装 fastdtw

只有 **stgode** 依赖 `fastdtw`，其它模型不需要。若不跑 stgode 可跳过。

```bash
uv add fastdtw
```

### 步骤 3：用 uv 跑脚本

下面所有命令都在 **TrafficFM 根目录**执行，用 `uv run` 保证用的是当前项目的环境：

```bash
# 一键跑所有实验模型的 eval_time
uv run python experiments/run_eval_time_all.py

# 只跑部分模型（不跑 stgode 可省掉 fastdtw）
uv run python experiments/run_eval_time_all.py --models agcrn,dcrnn,lstm,gwnet

# 单模型
uv run python experiments/agcrn/main.py --eval-time
```

### 若你更习惯用 conda

也可以新建一个 conda 环境，装齐依赖后再用该环境的 `python` 跑（不通过 uv）：

1. 创建环境：`conda create -n trafficfm_exp python=3.10 -y && conda activate trafficfm_exp`
2. 安装项目依赖：在 TrafficFM 根目录执行 `pip install -e .`（会装好 pyproject.toml 里的 gift_eval、chronos 等，间接带上 torch、gluonts）
3. 若要跑 stgode：`pip install fastdtw`
4. 运行：`python experiments/run_eval_time_all.py`（注意当前目录为 TrafficFM 根目录）

数据路径在运行时会自动按 TrafficFM 根目录修正，无需改 benchmark 的 config。

## 用法小结

**在 TrafficFM 项目根目录下执行**（不要 cd 到 experiments/），用 **uv run** 或已激活的 conda 环境里的 **python**：

```bash
cd TrafficFM

# 一键跑所有实验模型的 eval_time
uv run python experiments/run_eval_time_all.py

# 只汇总已有 time_estimation.json（不跑子进程）
uv run python experiments/run_eval_time_all.py --aggregate-only

# 只跑指定模型（逗号分隔）
uv run python experiments/run_eval_time_all.py --models agcrn,dcrnn,lstm
```

单模型：

```bash
uv run python experiments/agcrn/main.py --eval-time
```

## 输出

- 每个模型：`results/<model_name>/time_estimation.json`（与 benchmark 的 `time_estimation.json` 结构相同）
- 汇总：`results/experiments_time_summary.csv` 与 `results/experiments_time_summary.json`

## 与 benchmark 的对比

- **Benchmark**：`python benchmark/run_eval_time_all.py` → `results/foundation_models_time_summary.csv`
- **Experiments**：`python experiments/run_eval_time_all.py` → `results/experiments_time_summary.csv`

两边都使用同一套 eval_time 逻辑（相同数据集列表、相同 estimation_samples、相同 JSON 字段），因此可以直接对比 `total_seconds` / `total_hours` 等字段。

## 模型列表

当前参与 eval_time 的实验模型：agcrn, astgcn, d2stgnn, dcrnn, dgcrn, dstagnn, gwnet, hl, lstm, stgcn, stgode, sttn。
