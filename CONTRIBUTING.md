# 贡献指南

[English](CONTRIBUTING.en.md)

欢迎通过 Issue 或 Pull Request 帮助改进实验、修复代码、补充复现步骤和文档。提交前请先查看相关实验的 README 和已有 Issue、Pull Request；涉及新实验或较大的设计改动，建议先开 Issue 讨论问题和方案。中文和英文均可。

## 准备环境

1. Fork 并克隆仓库，为改动创建分支。
2. 阅读目标实验目录的 README，按其中的硬件、系统、依赖、模型和数据准备环境。各实验相互独立，仓库根目录的 `requirements.txt` 只包含开发工具，不会安装实验运行依赖。
3. 如需运行格式等开发检查，在独立的 Python 环境中安装开发工具：

   ```bash
   python -m pip install -r requirements.txt
   ```

4. 在普通克隆的仓库根目录安装仓库钩子：

   ```bash
   bash scripts/install-hooks.sh
   ```

   该脚本会覆盖 `.git/hooks/pre-commit` 中已有的钩子；如果已配置其他钩子，请先保留并自行合并。安装后，提交时会检查暂存文件的大小和 notebook 输出。Windows 可使用 Git Bash。

## 修改与验证

- 每个改动尽量聚焦一个问题。修复代码时补充能够验证修复的测试；文档改动检查命令、链接和相关描述。
- 新增或修改实验时，在实验 README 中写清适用系统与硬件、环境和依赖准备、模型或数据来源、运行命令、检查方法及已知限制。记录实测环境；最低要求或兼容性未经验证时请说明。实验讲解、结果和适用边界放在 `docs/` 中的对应文档。
- 按改动运行相关测试或实验的低成本检查，并在 PR 中说明实际运行的命令、结果与未覆盖的部分。不要求每次运行所有高成本训练和基准；没有运行的实验不要写成已验证。
- 模型权重、数据集、checkpoint 和大体积运行产物不要提交到 Git；提供来源或重建方法。需要保留的实验指标和证据应足以核对，文件大小限制见 [AGENTS.md](AGENTS.md)。保留第三方材料要求的署名、许可证和来源。
- Python 代码使用 Black，行宽 100。项目的 pre-commit 配置还包含空白字符、文件末尾和 YAML 检查；安装开发依赖后可手动运行：

  ```bash
  python -m pre_commit run --all-files
  ```

  此命令可能修改文件；请检查结果。仓库钩子只运行文件大小与 notebook 输出检查，不会自动运行这些格式检查。CI 还会检查 Git 历史打包体积，不会自动运行全部实验。

提交前可检查暂存内容：

```bash
python scripts/check_large_files.py
python scripts/check_notebook_outputs.py
git diff --cached --check
```

前两项默认检查暂存区；`--all` 检查 Git 已跟踪文件，不包含未跟踪的新文件。notebook 提交前请清除执行输出，必要的结果可另存为证据并注明复现方式。

## 提交 Pull Request

1. 使用清晰的提交信息，遵循 Conventional Commits，例如 `fix: 修正采样边界` 或 `docs: 补充环境准备步骤`。
2. 在 PR 中说明要解决的问题、主要改动、验证命令和结果；关联相关 Issue。实验结果请附上必要的设备、数据、配置和测量条件。
3. 标明尚未验证的部分、复现限制及依赖的外部资源；根据审阅反馈更新 PR。

代码使用 [Apache License 2.0](LICENSE)。提交第三方代码或材料时，请确保有权使用并遵守其适用许可。
