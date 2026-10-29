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

> Usage example. What's the minimal meaningful example that demonstrates the core ideas from your project?

## 📚 Citation

> Citation instructions. How should people cite your work?
