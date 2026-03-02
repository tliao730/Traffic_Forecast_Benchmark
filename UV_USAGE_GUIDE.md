## TrafficFM Project Guide for Environment Management with `uv`

This guide summarizes all `uv`-related concepts, commands, and recommended workflows we discussed, tailored for the `TrafficFM` project and running different models under `benchmark/`.

---

## 1. Core Concepts and Files

- **`pyproject.toml`**
  - Config file that declares project metadata and "what dependencies are needed".
  - Maintained by you or tools; it is the **source of truth**.

- **`uv.lock`**
  - Lock file that records all dependencies and their exact versions (including transitive dependencies).
  - Generated/updated automatically by `uv` for **environment reproducibility**.

- **Reproducibility Principle**
  - The same pair of `pyproject.toml` + `uv.lock` defines **one environment**.
  - In that environment, **each package can have only one version**.

---

## 2. Basic Workflow: Existing Project (TrafficFM)

In the `TrafficFM` root directory:

- **Create/Sync Environment**

  ```bash
  cd TrafficFM
  uv sync
  ```

  - Installs or updates dependencies according to the current `pyproject.toml` / `uv.lock`.
  - By default, it does an "exact sync" and **removes extra packages** in the environment that are not in the lock file.

- **Run Code**

  ```bash
  uv run benchmark/chronos_1.py
  ```

  - Uses the interpreter and dependencies from the project's virtual environment.

---

## 3. Ways to Install Dependencies and Their Differences

### 3.1 Add as Project Dependency (Recommended)

- **Add runtime dependency**

  ```bash
  uv add package_name
  # or pin a version
  uv add "package_name==1.2.3"
  ```

  Effect:
  - Modifies `pyproject.toml` (adds the dependency).
  - Resolves dependencies and updates `uv.lock`.
  - Installs/updates the package in the virtual environment.

- **Add dev dependency**

  ```bash
  uv add --dev pytest
  ```

### 3.2 Temporary Install (Not Written to `.toml` / `.lock`)

- **Similar to `pip install`**

  ```bash
  uv pip install xgboost
  ```

  Characteristics:
  - Installs the package in the current environment (usually the project's virtual environment).
  - **Does not modify `pyproject.toml` or `uv.lock`**.
  - Suitable for local ad-hoc experiments.

- **Add a package only for a single run**

  ```bash
  uv run --with xgboost benchmark/chronos_1.py
  ```

  Characteristics:
  - This run temporarily includes `xgboost` as a dependency.
  - Does not write to `pyproject.toml` / `uv.lock`; environment definition stays clean.

---

## 4. Restore Environment to Clean State

Scenario: You temporarily installed a package with `uv pip install xgboost` and now want the environment to return to a state that "contains only project-declared dependencies".

- **In the project root directory, run:**

  ```bash
  uv sync
  ```

Behavior:
- Syncs the environment exactly according to `uv.lock`:
  - Keeps all packages and versions listed in the lock file.
  - **Removes any extra packages** installed but not in the lock file (e.g., via `uv pip install`).
- Note: If you use `uv sync --inexact`, extra packages will not be removed, but the default `uv sync` does clean them.

---

## 5. Migrate from Temporarily Used Version to Officially Locked Version

Scenario: You installed a version of `xgboost` in `.venv` via `uv pip install`, verified it works, and want to record it in `pyproject.toml` and `uv.lock` for reproducibility.

Steps:

1. **Check the current version of this package in the environment**

   ```bash
   uv run python -m pip show xgboost
   # or
   uv run python -m pip list | grep xgboost
   ```

   Note the version, e.g. `2.0.3`.

2. **Add it as an official dependency with that version**

   ```bash
   uv add "xgboost==2.0.3"
   ```

   Result:
   - `pyproject.toml` gains `xgboost==2.0.3`.
   - `uv.lock` is updated and locks this version.
   - Anyone running `uv sync` will get `xgboost==2.0.3`.

---

## 6. Managing Multiple Environments / Multiple Models

### 6.1 Constraints in a Single Environment

- One `pyproject.toml` + `uv.lock` corresponds to **one environment**.
- In that environment:
  - **Each package can have only one version**.
  - You cannot use `[project.optional-dependencies]` to lock both:

    ```toml
    [project.optional-dependencies]
    model_a = ["xgboost==2.0.3"]
    model_b = ["xgboost==2.1.3"]
    ```

    This would cause a dependency resolution conflict.

### 6.2 Standard Approach for Multiple Environments

To support different models with incompatible package versions, use **multiple separate uv projects**:

- Each project directory contains:
  - `pyproject.toml`
  - `uv.lock`
  - Its own virtual environment

Without changing the `benchmark/` directory layout, you can add an `envs/` directory in the repo, e.g.:

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

Usage example:

**Recommended: Run from `benchmark/` with `--project` to specify the model environment** (working directory stays `benchmark/`, results are written correctly to `TrafficFM/results/`):

  ```bash
  cd TrafficFM/benchmark
  uv run --project ../envs/arima python arima.py
  uv run --project ../envs/moirai python chronos_1.py
  uv run --project ../envs/flowstate python feedforward.py
  uv run --project ../envs/flowstate python flowstate.py
  uv run --project ../envs/kairos python kairos.py
  uv run --project ../envs/moirai python moirai.py
  uv run --project ../envs/moirai python moirai2.py
  uv run --project ../envs/flowstate python naive.py
  uv run --project ../envs/sundial python sundial.py
  uv run --project ../envs/tabpfn_ts python tabpfn_ts.py
  uv run --project ../envs/timesfm_eval python timesfm_eval.py
  uv run --project ../envs/toto python toto.py
  uv run --project ../envs/tabpfn_ts python ml_ensemble.py
  uv run --project ../envs/tabpfn_ts python ml_methods.py
  ```

  This way:
  - The current directory is `benchmark/`, so relative paths in scripts (e.g. `../results`) resolve to `TrafficFM/results` instead of `envs/results`.
  - `--project ../envs/model_name` specifies which model's uv environment (dependencies and .venv) to use.

**Not recommended: cd into the env first, then run the script** (working directory becomes the env directory; `../results` would point to `envs/results`):

  ```bash
  cd TrafficFM/envs/chronos
  uv sync
  uv run ../../benchmark/chronos_1.py   # Results would be written to envs/results/
  ```

Result:
- Scripts under `benchmark/` stay in place.
- Each model has its own independent `pyproject.toml` / `uv.lock` and virtual environment, so you can use different and incompatible dependency versions freely.

---

## 7. Migrating from `requirements.txt` to uv Project

If an environment was originally managed with `requirements.txt`, you can migrate to uv project mode:

```bash
# Create a minimal pyproject.toml in the target directory (which will be the new project root)
uv init --bare

# Import dependencies from requirements.txt and generate/update uv.lock
uv add -r requirements.txt
```

After that:
- Use `uv add` / `uv remove` / `uv sync` to manage dependencies.
- `pyproject.toml` + `uv.lock` replace `requirements.txt` as the source of environment definition and locking.

---

## 8. Recommended Practices Summary

- **Official dependencies**: Use `uv add` to write to `pyproject.toml` and let `uv.lock` pin versions.
- **Temporary experiments**: Use `uv pip install` or `uv run --with package_name`; convert to official dependencies as needed after testing.
- **Restore environment**: To return to a clean state at any time, run `uv sync` in the project root.
- **Multiple incompatible environments**: Use multiple directories (e.g. `envs/model_name/`), each with its own `pyproject.toml` + `uv.lock`. **Recommended: run from `benchmark/`** with `uv run --project ../envs/model_name python script.py`, so the working directory is `benchmark/` and results go to `TrafficFM/results/`.

