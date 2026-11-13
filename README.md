# 📦 CCAI Repository Template

Template for new Python project repositories at the Centre for Credible AI.

It serves two purposes:

1. **Unification and best practices sharing.** We want to have consistent, well-structured, high-quality repositories which will represent the Centre well.
1. **Get you up to speed faster.** This template comes with a number of preconfigured tools so that you can start your research faster! We'll add more tools as needed. 

## 🚀 How to use this template

### ✨ Create a new repository from this template

1. Click "Use this template" ↗️.

1. Use [kebab-case](https://en.wikipedia.org/wiki/Naming_convention_(programming)#Examples_of_multiple-word_identifier_formats) naming convention for repository name.
1. Leave the **Owner** and **Visibility** options to their default values (CredibleAI and Private).

### 🛠️ Initialize your new repository

1. Clone your new repository.
2. In `pyproject.toml`, change `name = "project"` to your project name.
3. Install `uv`, the Python package manager: [instructions](https://docs.astral.sh/uv/getting-started/installation/).
4. Run the following commands:
   ```sh
   uv sync
   uv run pre-commit install
   ```
   They have the following effects:
   * Creates a local Python environment as well as a `uv.lock` file. ⚠️ Important: You must commit and push this `uv.lock` file in your very first commit. The repository includes a strict GitHub Actions CI pipeline that requires this lockfile. If you push without it, your first GitHub Action run will fail!
   * Configures Ruff's linter and formatter in pre-commit.
5. Start coding!

### 🔧 Optional configuration steps

1. Delete dependencies you don't need in `pyproject.toml`.

2. Remember to run `uv sync` again and commit the updated `uv.lock` if you change any dependencies.

## 📦 What does this template contain?

1. **Unified** project naming, `README.md` and folder structure.

   **Why:** Our repositories are consistent, so they're easier to understand.
2. **[uv](https://docs.astral.sh/uv/) package manager** with modern configuration.

   **Why:** Dependencies install in seconds. Ensures reproducibility (everyone using this repo has the exact same environment).
3. Most common **Python package dependencies**. 

   **Why:** You can start your development faster — the most common ML/research packages are pre-installed. (Feel free to delete what you don't need.)
4. **[Ruff](https://docs.astral.sh/ruff/) linter and formatter** configured in [pre-commit](https://pre-commit.com/).

5. **[GitHub Actions](https://docs.github.com/en/actions/get-started/understand-github-actions)** CI pipeline for automatic linting and type-checking.

   **Why:** Every push and pull request to `main` automatically runs Ruff and Ty in a strict locked environment. This guarantees code quality and catches errors before they are merged.

   **Why:** Automatic formatting and lint checks on every commit — consistent style and fewer bugs.
6. **[Setuptools](https://setuptools.pypa.io/en/latest/)** configuration. 

   **Why:** Your `src/` code is importable anywhere in the project — no path hacks needed.

## 📁 Folder structure

* `src/` — source folder. Everything you put here will be available as modules across the entire project. For example, a file `src/myproject/metrics.py` can be imported as `import myproject.metrics`.
* `notebooks/` — folder for notebooks.

## 🔄 Life of the repository

1. Please keep your repository as "Private" during development.
2. When your project is finished and ready to show to the world (congrats btw!):
   * Fill in the `README.md` template below.
   * Delete those instructions.
   * Change repository visibility to "Public".

💡 Have suggestions for improving this template? Submit a PR!

---

# [Project name]

> Project overview. What should a user know to understand what your project is about?

## 🛠️ Setup

> Setup instructions. What must a user do to be able to run your code?

## 🚀 How to use it

### Running the experiment suite

```sh
uv run python -m src.scripts.train_orchestrator --config configs/experiment_config.yaml
```

For every dataset and outer fold the orchestrator runs Stage 1 HPO (`single_resnet` backbone),
Stage 2 HPO for each configured method, and refits every method on the outer training
partition with each of `refit_seeds`, evaluating on the untouched outer test fold.
OpenML datasets listed in `data/links.csv` are downloaded on first use to `data/raw/`.

Useful options:

* `--datasets california pol` / `--models sfo_cka tangos` – restrict the run's scope.
* `--stages hpo` or `--stages refit` – run only one phase (refits need finished studies).
* `--fresh` – move the existing run directory aside and start from scratch.

`configs/smoke_config.yaml` runs the whole pipeline on the toy CSVs in `datasets/` in about
half a minute on CPU.

### Resuming after a crash or time limit

Re-run the same command. Everything under `results/<run_name>/` is resumable:

* `optuna.db` – all studies (`<dataset>/fold<k>/<method>`); only missing trials are run and an
  interrupted trial is re-run with the same parameters.
* `progress.csv` – finished refits, which are skipped.
* `checkpoints/` – training state of an interrupted refit (every `checkpoint.every_n_epochs`).

On `SIGTERM`, `SIGUSR1` (e.g. SLURM `--signal=USR1@300`), the first `Ctrl+C`, or when
`checkpoint.time_budget_hours` is exhausted, the run finishes the current epoch, saves a
checkpoint and exits with code 3, so a job script can resubmit itself while the exit code is 3.
A resumed run produces the same numbers as an uninterrupted one. The run refuses to start if
result-affecting settings (`training`, `hpo`, seeds, dataset definitions) differ from those
stored in the run directory; use a new `run_name` or `--fresh` in that case.

### Outputs (`results/<run_name>/`)

* `metrics.csv` – test metrics per dataset, fold, method and seed (accuracy, ROC-AUC, NLL or
  RMSE, MAE, R²; plus `n_params`, `epochs`, `train_seconds`). Read it with
  `src.run_tools.resume.load_results`, which drops rows duplicated by a crash.
* `diagnostics.csv` – staged ensemble performance and member output norms.
* `<dataset>/best_params/`, `<dataset>/similarity/`, `<dataset>/splits.npz`, `config.yaml`.
* `logs/<timestamp>_<SLURM job id or pid>.log` – copy of the console log of every invocation.

### Debugging cluster runs

Console output is timestamped and flushed line by line, so it appears in the SLURM output file
immediately. Useful tools when a job seems to do nothing:

* `-v` logs every trial (parameters, result, duration) and every refit; `-vv` additionally
  logs every epoch (train/validation loss, epoch time), progress inside epochs longer than
  a minute, and checkpoint saves.
* A heartbeat line is logged every 10 minutes (`--heartbeat-minutes`, 0 disables) with the
  current stage (dataset, fold, phase, method, trial or seed, epoch), the time since that stage
  last changed, and memory use. It comes from a background thread, so it keeps appearing while
  the main thread is blocked: a stage that stops changing while "last update" keeps growing
  means a hang, not slow training.
* `scancel --signal=USR2 <jobid>` (or `kill -USR2 <pid>`) prints the Python stack of every
  thread to stderr without stopping the run, showing exactly where it is stuck.
  `--dump-stacks-minutes N` does this automatically every N minutes.
* The startup lines report host, job id, library versions, device, GPU name, and torch/OpenMP
  thread counts. Set `OMP_NUM_THREADS` to `SLURM_CPUS_PER_TASK` on CPU nodes to avoid
  oversubscription.

Common causes of a silent hang:

* **No internet on compute nodes.** An OpenML dataset missing from `data/raw/` is downloaded on
  first use and the download can block indefinitely. Run
  `uv run python -m src.scripts.train_orchestrator --config ... --stages data` on a login node
  first; it only downloads/caches the datasets and writes the splits. The log says
  `using cached ...` or `not cached, downloading ...` for every dataset.
* **`uv run` syncing the environment** without network access. Sync on the login node and use
  `uv run --no-sync` (or `.venv/bin/python`) in the job.
* **SQLite on a network filesystem.** `-v` logs how long opening each Optuna study takes; if
  it is slow or never finishes, point `results_dir` to a local or scratch filesystem.

Example `sbatch` script that stops gracefully 5 minutes before the time limit and resubmits
itself until the suite is finished:

```sh
#!/bin/bash
#SBATCH --job-name=el-experiments
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@300
#SBATCH --output=slurm-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# srun puts Python in a job step so that it receives the USR1 signal itself.
srun uv run --no-sync python -m src.scripts.train_orchestrator \
    --config configs/experiment_config.yaml -v
status=$?
if [ "$status" -eq 3 ]; then
    sbatch "$0"
fi
exit "$status"
```

## 📚 Citation

> Citation instructions. How should people cite your work?
