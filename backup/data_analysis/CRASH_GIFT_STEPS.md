# 方案 A：Crash 数据构建 GIFT (Arrow) 数据集 — 分步执行说明

## 任务回顾

- **数据**：`crash_flow_after_2019-10-20.json`（每条 769×6，前 720 步 history，第 721–732 步为预测目标 12 步）
- **设定**：单变量，只用 crash_id（第 0 列）；history=720，预测 future=12；评估 MAE/MSE
- **目标**：构建 Arrow 数据集 → 用现有 benchmark（如 Moirai2、Chronos）跑 eval，得到整体 MAE/MSE

---

## 已完成的代码/配置修改

以下已在仓库中改好，无需你再改：

1. **`data_analysis/generate_crash_gift_eval.py`**  
   从 crash JSON 生成 GIFT 格式 Arrow 数据集（每条序列 720+12，item_id/start/freq/target）。

2. **`benchmark/dataset_properties.json`**  
   已增加 `"crash_sd"` 条目（frequency 15T, domain Transport, num_variates 1）。

3. **`benchmark/common.py`**  
   在 `setup_dataset()` 中已为 `crash_sd` 增加与 ca/gba/gla 相同的 fallback 注册。

4. **`benchmark/config.py`**  
   已在 `short_datasets` 和 `med_long_datasets` 中加入 `crash_sd/2019/15T`。

---

## 你需要执行的步骤（按顺序）

### 步骤 1：确认环境与路径

- 建议在 **TrafficFM 项目根目录** 下执行后续命令（即包含 `crash_flow_after_2019-10-20.json`、`data_analysis/`、`benchmark/` 的目录）；脚本也会把相对路径的 `--input`/`--output_dir` 解析为相对项目根，从其他目录运行也可。
- 确保已安装依赖：`datasets`（HuggingFace）、`pandas`、`numpy` 等（与现有 `generate_data_for_gift_eval.py` / benchmark 一致即可）。

### 步骤 2：生成 Crash 的 Arrow 数据集

在项目根目录执行：

```bash
python data_analysis/generate_crash_gift_eval.py \
  --input crash_flow_after_2019-10-20.json \
  --output_dir ./data/gift_eval_datasets \
  --overwrite
```

- `--input`：crash JSON 路径；若文件在项目根目录，上述路径即可。
- `--output_dir`：必须与 benchmark 使用的 GIFT 根目录一致；当前 benchmark 的 `gift_eval_datasets_path` 为 `../data/gift_eval_datasets`（相对 benchmark 目录），因此这里用 `./data/gift_eval_datasets`。
- `--overwrite`：若已存在 `data/gift_eval_datasets/crash_sd/2019/15T` 会先删再写。

成功后会看到类似：

- `Saved N series to .../data/gift_eval_datasets/crash_sd/2019/15T`
- `Dataset name for benchmark: crash_sd/2019/15T`
- `Use term='long' for prediction_length=12.`

可选：若希望每个 crash 的 6 个 ID 各成一条序列，可加 `--all_ids`（样本数会变为约 1457×6）。

### 步骤 3：确认 GIFT 数据目录与 benchmark 一致

- 确认 `benchmark/config.py` 里 `gift_eval_datasets_path` 指向的目录就是你在步骤 2 用的 `output_dir`。
- 当前为 `../data/gift_eval_datasets`（相对 `benchmark/`），即项目下的 `data/gift_eval_datasets`。  
  若你实际 GIFT 数据在别处，请同时：
  - 在步骤 2 用 `--output_dir <你的GIFT根目录>`
  - 在 `benchmark/config.py` 中把 `gift_eval_datasets_path` 改为同一路径（或对应相对路径）。

### 步骤 4：只跑 Crash 数据集（可选：仅 long term）

当前 `short_datasets` 和 `med_long_datasets` 里都包含 `crash_sd/2019/15T`，因此跑全量 benchmark 时会自动包含 crash，并跑 short/medium/long 三个 term。  
你只需要 **prediction_length=12**，即 **term=long**；若想先只验证 crash 的 long：

- 可临时把 `benchmark/config.py` 改为仅包含 crash，例如：
  - `short_datasets = "crash_sd/2019/15T"`
  - `med_long_datasets = "crash_sd/2019/15T"`
- 然后运行对应模型（见步骤 5），这样只会对 crash_sd 跑 short/medium/long，其中 long 对应 12 步。

验证通过后，再根据需要改回包含 sd/gba/gla/ca 等。

### 步骤 5：运行 Foundation Model 得到 MAE/MSE

在 **`benchmark/`** 目录下执行（与现有 GIFT 用法一致）：

```bash
cd benchmark
python moirai2.py
```

或 Chronos：

```bash
python chronos_1.py
```

- 脚本会调用 `common.eval()`，自动遍历 `config` 里的数据集（含 `crash_sd/2019/15T`）和 term（short/medium/long）。
- 结果会写入 `../results/<model_name>/all_results.csv`（或当前 config 指定的 result 目录），其中会包含 `crash_sd/15T/long`（或类似）的 MAE/MSE。

只看 **prediction_length=12** 的指标时，在结果 CSV 里筛选 **term=long** 或 **prediction_length=12** 对应的行即可。

---

## 小结

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 在项目根目录、确认环境 | 依赖与现有 GIFT/benchmark 一致 |
| 2 | 运行 `generate_crash_gift_eval.py` | 生成 `data/gift_eval_datasets/crash_sd/2019/15T` |
| 3 | 确认 `output_dir` 与 config 中 GIFT 路径一致 | 避免 benchmark 找不到数据集 |
| 4 | （可选）临时只保留 crash 在 config | 便于只验证 crash、long term |
| 5 | 在 `benchmark/` 下运行 `moirai2.py` 或 `chronos_1.py` | 得到含 crash 的 MAE/MSE，long=12 步 |

**benchmark 下的模型代码（如 `moirai2.py`、`chronos_1.py`、`common.py` 的 eval 逻辑）无需改动**；只要数据集名 `crash_sd/2019/15T` 在 config 的列表里且 Arrow 数据已生成，就会自动参与评估。
