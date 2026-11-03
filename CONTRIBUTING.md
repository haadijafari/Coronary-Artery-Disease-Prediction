# 🤝 Contributing Guidelines

Contributions are always welcome!

To maintain a clean and readable history, we strictly follow a specific commit message convention. Please ensure your contributions adhere to the following structure.

## ✔️ Commit Guide

### The Structure

Every commit must consist of a **Header** and a **Body**, separated by an empty line.

```Plaintext
[TAG] (app_name) Capitalized short summary (50 chars or less)

[+] Detail about what was added
[-] Detail about what was removed
[*] Detail about what was changed/refactored
```

1. **The Header Rules**
    - **Imperative Mood:** Use "Add", "Fix", "Change" (not "Added", "Fixed", "Changed"). Think of the commit as an instruction.

    - **App Name:** mention the app name your commit is touching inside parenthesis. This project uses the following scopes:

      | Scope     | Covers                                                        |
      |-----------|---------------------------------------------------------------|
      | `core`    | Project setup, packaging, tooling, CI, repository housekeeping |
      | `data`    | Raw data, loading, cleaning and validation                     |
      | `eda`     | Exploratory analysis, plotting helpers, figures                |
      | `model`   | Preprocessing, models, evaluation, training, saved artifacts   |
      | `app`     | The Streamlit application                                      |
      | `docs`    | README, contributing guide and other documentation             |

    - **Allowed Tags:** Start your commit with one of the following:

      - `[INIT]`: Initializing a project or module.

      - `[ADD]`: Adding new features or files.

      - `[DEL]`: Removing files or features.

      - `[FIX]`: Fixing a bug.

      - `[REFACTOR]`: Changing code structure without changing behavior.

      - `[PERF]`: Improving performance.

      - `[STYLE]`: Formatting, missing semi-colons, etc. (no code change).

      - `[CHORE]`: Updating build tasks, package manager configs, etc.

2. **The Body Rules**

The body should list the specific changes made. Use the following bullets to categorize your changes:

- `[+]`: Used for creating files or adding significant code blocks.

- `[-]`: Used for deleting files or removing code.

- `[*]`: Used for modifying existing logic or file content.

### Example

#### ❌ Bad Commit

```Plaintext
fixed cleaning and added some tests
```

#### ✅ Good Commit

```Plaintext
[FIX] (data) Treat sentinel codes as missing values

[+] Add tests covering caa=4 and thall=0 rows
[*] Map sentinel codes to NaN before decoding categories
[*] Update data dictionary in README.md
```

## 🌿 Branching Strategy

We use a strict naming convention to keep track of features and fixes based on the application scope.

1. **Main Branch**
    - `main`: always stable. Every commit on `main` passes CI (lint, tests and a full notebook run), and the Streamlit app is deployed from it. Never commit to `main` directly; all work lands through a branch.

2. **Naming Your Branch**
Format: `type/app_name-description`

    |Type     |Use Case                         |Example                            |
    |---------|---------------------------------|-----------------------------------|
    |`feat/`  |New features, Inits, Performance |`feat/model-threshold-tuning`      |
    |`fix/`   |Bug fixes, Refactoring, Style    |`fix/data-sentinel-values`         |
    |`hotfix/`|Urgent fixes to the deployed app |`hotfix/app-broken-input-form`     |
    |`chore/` |Build, config, or auxiliary tools|`chore/core-update-dependencies`   |

3. **Workflow**
    1. Switch to `main` and pull the latest changes: `git switch main && git pull`.

    2. Create your branch: `git switch -c feat/model-cool-feature`.

    3. Commit your work using the [Commit Guidelines](#️-commit-guide).

    4. Run the checks locally before pushing:

        ```bash
        uv run ruff check . && uv run ruff format --check . && uv run pytest
        ```

    5. Rebase onto the latest `main`, push, and open a Pull Request to `main`. Merge with **Rebase and merge** so the history stays linear.
