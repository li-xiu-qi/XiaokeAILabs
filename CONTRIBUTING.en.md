# Contributing

[中文](CONTRIBUTING.md)

Issues and pull requests are welcome for experiment improvements, bug fixes, reproduction steps, and documentation. Before starting, read the relevant experiment README and check existing issues and pull requests. For a new experiment or a substantial design change, consider opening an issue first to discuss the problem and approach. Contributions in English or Chinese are welcome.

## Set up

1. Fork and clone the repository, then create a branch for your changes.
2. Follow the target experiment's README for its hardware, system, dependencies, models, and data. Experiments have separate environments; the root `requirements.txt` contains development tools only.
3. To run development checks, install those tools in a separate Python environment:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. In the root of a regular clone, install the repository hook:

   ```bash
   bash scripts/install-hooks.sh
   ```

   This script overwrites an existing `.git/hooks/pre-commit` hook. Preserve and merge any existing hook first. The installed hook checks staged file sizes and notebook outputs. On Windows, Git Bash can run the script.

## Make and verify changes

- Keep each change focused. Add tests that demonstrate a code fix, and check commands, links, and related descriptions for documentation changes.
- For new or modified experiments, document the supported system and hardware, environment and dependency setup, model or data sources, commands, checks, and known limitations in the experiment README. Record the tested environment and identify minimum requirements or compatibility claims that have not been verified. Put explanations, results, and their limits in the corresponding `docs/` document.
- Run tests or low-cost checks relevant to the change. In the PR, report the commands, results, and any gaps. Running every expensive training job or benchmark is not required; do not claim an experiment was verified if it was not run.
- Do not commit model weights, datasets, checkpoints, or large runtime outputs to Git. Provide sources or reconstruction steps instead. Keep enough metrics and evidence to check experiment results; see [AGENTS.md](AGENTS.md) for file-size limits. Preserve required attribution, licenses, and sources for third-party material.
- Format Python code with Black at line length 100. The pre-commit configuration also checks whitespace, end-of-file, and YAML. After installing development dependencies, run it manually with:

  ```bash
  python -m pre_commit run --all-files
  ```

  This command can modify files; review its changes. The repository hook only checks file size and notebook outputs; it does not automatically run these formatting checks. CI also checks the packed size of Git history; it does not run all experiments.

Before committing, you can check the staged contents:

```bash
python scripts/check_large_files.py
python scripts/check_notebook_outputs.py
git diff --cached --check
```

By default, the first two commands check the index. Their `--all` option checks Git-tracked files, not new untracked files. Clear notebook outputs before committing; preserve necessary results separately as evidence and document how to reproduce them.

## Open a pull request

1. Use clear Conventional Commits messages, such as `fix: correct sampling boundary` or `docs: add environment setup`.
2. Describe the problem, changes, validation commands, and results in the PR, and link relevant issues. For experiment results, include the device, data, configuration, and measurement conditions needed to interpret them.
3. Identify unverified parts, reproduction limits, and external resources; update the PR in response to review.

The code is available under the [Apache License 2.0](LICENSE). Make sure you have the right to submit third-party code or material and comply with its applicable license.
