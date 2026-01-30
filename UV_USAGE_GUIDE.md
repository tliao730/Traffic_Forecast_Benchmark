## TrafficFM 项目下使用 `uv` 的环境管理指南

本指南总结了我们前面讨论的所有 `uv` 相关概念、命令和推荐工作流，特别适用于 `TrafficFM` 项目和你在 `benchmark/` 下运行不同模型时的需求。

---

## 1. 核心概念与文件

- **`pyproject.toml`**
  - 声明项目元数据和「需要哪些依赖」的配置文件。
  - 由你或工具维护，是**源配置**。

- **`uv.lock`**
  - 记录所有依赖及其精确版本（包括子依赖）的锁文件。
  - 由 `uv` 自动生成/更新，用于**环境复现**。

- **环境复现原则**
  - 同一对 `pyproject.toml` + `uv.lock` 定义**一套环境**。
  - 在这套环境中，**同一个包只能有一个版本**。

---

## 2. 基本工作流：已有项目（TrafficFM）

在 `TrafficFM` 根目录：

- **创建/同步环境**

  ```bash
  cd TrafficFM
  uv sync
  ```

  - 根据当前的 `pyproject.toml` / `uv.lock` 安装或更新依赖。
  - 默认是「精准同步」，会移除环境中**不在锁文件里的多余包**。

- **运行代码**

  ```bash
  uv run benchmark/chronos_1.py
  ```

  - 使用项目对应虚拟环境中的解释器和依赖。

---

## 3. 安装依赖的几种方式与差异

### 3.1 正式加入项目依赖（推荐）

- **添加运行依赖**

  ```bash
  uv add 包名
  # 或固定版本
  uv add "包名==1.2.3"
  ```

  作用：
  - 修改 `pyproject.toml`（添加依赖）。
  - 重新解算依赖并更新 `uv.lock`。
  - 安装/更新虚拟环境中的包。

- **添加开发依赖**

  ```bash
  uv add --dev pytest
  ```

### 3.2 临时安装（不写入 `.toml` / `.lock`）

- **类似 `pip install` 的方式**

  ```bash
  uv pip install xgboost
  ```

  特点：
  - 在当前环境（通常是项目的虚拟环境）里安装包。
  - **不会修改 `pyproject.toml` 和 `uv.lock`**。
  - 适合本地临时试验。

- **只在单次运行中额外带上某个包**

  ```bash
  uv run --with xgboost benchmark/chronos_1.py
  ```

  特点：
  - 本次运行临时加入 `xgboost` 作为依赖。
  - 不写入 `pyproject.toml` / `uv.lock`，环境定义不被污染。

---

## 4. 恢复环境到干净状态

场景：你用 `uv pip install xgboost` 临时装了一个包，现在想让环境回到「只包含项目声明依赖」的状态。

- **在项目根目录执行：**

  ```bash
  uv sync
  ```

行为：
- 以 `uv.lock` 为标准，精准同步环境：
  - 保留锁文件中列出的所有包和版本。
  - **移除所有额外安装但不在锁文件中的包**（例如通过 `uv pip install` 临时装的）。
- 注意：如果你显式使用了 `uv sync --inexact`，则不会移除“多余包”，但默认 `uv sync` 是会清理的。

---

## 5. 从临时可用版本迁移到正式锁定版本

场景：你通过 `uv pip install xgboost` 在 `.venv` 里装了一个版本，测试后确认这个版本是可用的，想把它写入 `pyproject.toml` 和 `uv.lock` 以便后续复现。

步骤：

1. **查当前环境中这个包的版本**

   ```bash
   uv run python -m pip show xgboost
   # 或
   uv run python -m pip list | grep xgboost
   ```

   记下版本号，例如 `2.0.3`。

2. **用该版本号正式加入依赖**

   ```bash
   uv add "xgboost==2.0.3"
   ```

   效果：
   - `pyproject.toml` 中增加 `xgboost==2.0.3`。
   - `uv.lock` 更新并锁定这个版本。
   - 之后任何人 `uv sync` 都会使用 `xgboost==2.0.3`。

---

## 6. 多个环境 / 多个模型的管理思路

### 6.1 同一个环境中的限制

- 一个 `pyproject.toml` + `uv.lock` 对应**一套环境**。
- 在这一套环境里：
  - **同一个包只能有一个版本**。
  - 不能通过 `[project.optional-dependencies]` 同时锁定：

    ```toml
    [project.optional-dependencies]
    model_a = ["xgboost==2.0.3"]
    model_b = ["xgboost==2.1.3"]
    ```

    这会在求解依赖时产生冲突。

### 6.2 多套环境的标准做法

要支持不同模型使用不兼容的包版本，需要**多套独立的 `uv` 项目**：

- 每个项目目录各自包含：
  - `pyproject.toml`
  - `uv.lock`
  - 自己的虚拟环境

在不修改 `benchmark/` 目录结构的前提下，可以在仓库中新增一个 `envs/` 目录，例如：

```text
TrafficFM/
  benchmark/
    chronos_1.py
    model_x.py
    ...

  envs/
    chronos/
      pyproject.toml
      uv.lock
    model_x/
      pyproject.toml
      uv.lock
```

用法示例：

**推荐：在 `benchmark/` 下执行，用 `--project` 指定模型环境**（工作目录保持为 `benchmark/`，结果会正确写到 `TrafficFM/results/`）：

  ```bash
  cd TrafficFM/benchmark
  uv run --project ../envs/kairos python kairos.py
  uv run --project ../envs/flowstate python flowstate_new.py
  uv run --project ../envs/feedforward python feedforward_new.py
  ```

  这样：
  - 当前目录是 `benchmark/`，脚本里的相对路径（如 `../results`）会解析到 `TrafficFM/results`，不会写到 `envs/results`。
  - `--project ../envs/模型名` 指定用哪个模型的 uv 环境（依赖和 .venv）。

**不推荐：先 cd 到 env 再跑脚本**（工作目录变成 env 目录，`../results` 会变成 `envs/results`）：

  ```bash
  cd TrafficFM/envs/chronos
  uv sync
  uv run ../../benchmark/chronos_1.py   # 结果会写到 envs/results/
  ```

这样：
- `benchmark/` 下的脚本位置不变。
- 每个模型对应一套完全独立的 `pyproject.toml` / `uv.lock` 和虚拟环境，可自由使用不同且互不兼容的依赖版本。

---

## 7. 从 `requirements.txt` 迁移到 `uv` 项目

如果某个环境原本是用 `requirements.txt` 管理的，可以迁移到 `uv` 项目模式：

```bash
# 在目标目录（将作为新项目根）创建最小 pyproject.toml
uv init --bare

# 从 requirements.txt 导入依赖并生成/更新 uv.lock
uv add -r requirements.txt
```

之后：
- 使用 `uv add` / `uv remove` / `uv sync` 管理依赖。
- `pyproject.toml` + `uv.lock` 取代 `requirements.txt` 作为环境定义和锁定的来源。

---

## 8. 推荐实践总结

- **正式依赖**：用 `uv add` 写入 `pyproject.toml`，让 `uv.lock` 锁定版本。
- **临时试验**：用 `uv pip install` 或 `uv run --with 包名`，测试通过后再按需转换为正式依赖。
- **恢复环境**：任何时候想回到干净状态，在项目根目录执行 `uv sync`。
- **多个互不兼容的环境**：使用多个目录（例如 `envs/模型名/`），每个目录一对 `pyproject.toml` + `uv.lock`。**推荐在 `benchmark/` 下执行**：`uv run --project ../envs/模型名 python 脚本.py`，这样工作目录是 `benchmark/`，结果会写到 `TrafficFM/results/`。

