# 基于规则与矢量指令流的 PDF 解析实测

[English](README.en.md)

本目录包含传统基于规则与原生指令流（PDF Vector / Content Stream）的文档提取实验，评估主流框架在权威开源测试用例上的性能、耗时、表格还原与物理边界。

[scripts](scripts/) · [历史证据](evidence/)

## 一、 测试样例来源（知名开源项目官方样本）

为确保测试的真实性与权威性，测试文件直接选用知名开源项目官方测试集的代表性样本：
1. **`rapidocr_direct_and_image.pdf`**（来自 `RapidAI/RapidOCRPDF` 官方测试集，703 KB）：
   包含数字原生文本流与嵌入图像区域的典型混合 PDF；
2. **`pdfplumber_federal_register.pdf`**（来自 `jsvine/pdfplumber` 官方测试集，697 KB）：
   长达 15 页的复杂多栏美国联邦公报（Federal Register），包含 60,000 余字、密集排版、线条表格与细密脚注；
3. **`testpage_cn.pdf`**（285 KB）：
   典型的扫描件包装 PDF，底层为纯位图渲染包装，用于验证规则解析器的零召回致盲边界。

---

## 二、 参评的主流传统框架

1. **`pypdf`**：纯 Python 实现，零外部依赖，极速轻量；
2. **`PyMuPDF (fitz)`**：基于 C/C++ MuPDF 高性能引擎，支持文本流、图像提取与原生表格检测（`find_tables`）；
3. **`pdfplumber`**：基于 pdfminer 的视觉与表格抽取高级包装库，专精于基于矢量线与矩形的精确表格提取；
4. **`pdfminer.six`**：深入字符级排版与绝对物理坐标的流解析器。

---

## 三、 复现命令

以下命令从实验目录执行。历史环境为 Windows、Python 3.13，规则解析可使用 CPU。新环境可先创建虚拟环境，然后安装解析依赖；完整版本未锁定，不保证重现历史数值：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate；Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-rules.txt
python -m pip check
python -c "import pymupdf, pypdf, pdfplumber, pdfminer; print('PDF 解析依赖可导入')"
```

2026-11-13 在 Linux ARM64、Python 3.12.3 的新虚拟环境中，按上述清单安装后通过 `pip check`；当次版本为 PyMuPDF 2.1、pypdf 7.0.1、pdfplumber 0.11.10 和 pdfminer.six 20260107。合成 PDF 脚本成功运行，两个解析器均能提取单栏与双栏样例的文字，扫描图像包装页均未提取到文字。此次仅验证合成样例的运行，不代表真实样本、OCR 或历史结果已经复现；清单未锁定，后续安装版本可能不同。

规则解析的合成 PDF 脚本自行生成输入；真实样例需放入 `data/reference_samples/`。上游可核对的候选为 [RapidOCRPDF 的 direct_and_image.pdf](https://github.com/RapidAI/RapidOCRPDF/blob/main/tests/test_files/direct_and_image.pdf) 和 [pdfplumber 的 Federal Register 测试 PDF](https://github.com/jsvine/pdfplumber/blob/stable/tests/pdfs/federal-register-2020-17221.pdf)，分别保存为 `rapidocr_direct_and_image.pdf` 和 `pdfplumber_federal_register.pdf`。历史记录未保存其来源 revision 或完整哈希，当前文件不能保证与历史样例相同。`testpage_cn.pdf` 的历史来源及发布条件尚未核实，需自行提供合法扫描 PDF；替代文件的结果不与历史结果直接比较。缺少样例时不能视为完成对应测量。

2026-11-14 使用上述依赖组合的独立环境，在 Linux ARM64 上分别调用 `benchmark_single_file` 处理了两份固定版本的上游样本。四个解析器均成功处理 2 页的混合 PDF 和 15 页的 Federal Register PDF；这只验证两个文件的规则解析，不代表扫描件、OCR、unstructured 或历史整套基准已复现。运行环境由已安装环境复制，未重复从零安装；单次耗时不作为性能结论。

以下命令在独立工作副本中获取本次使用的两份文件，并核对 SHA-256；另一个扫描 PDF 仍需自行准备，才能运行后面的完整三文件命令：

```bash
mkdir -p data/reference_samples
curl -fL https://raw.githubusercontent.com/RapidAI/RapidOCRPDF/bb99e81e397c53fc9b6bba71d49408470c02f09e/tests/test_files/direct_and_image.pdf -o data/reference_samples/rapidocr_direct_and_image.pdf
curl -fL https://raw.githubusercontent.com/jsvine/pdfplumber/4c64b92d5caccd71c645e98e0fabb0c4dba7ff45/tests/pdfs/federal-register-2020-17221.pdf -o data/reference_samples/pdfplumber_federal_register.pdf
sha256sum data/reference_samples/rapidocr_direct_and_image.pdf data/reference_samples/pdfplumber_federal_register.pdf
# 分别应为 9823477e6d872d46f9bd7b5bff75092d49b097841919eccb9855a0aa06d5ebf9
#         ac419e4e084ea7a7a33a7be3f129dd07bc705b1a9e45ad751e33357f17d60f5c
```

脚本会在实验目录下生成 `data/` 和 `results/`，并可能覆盖或因已有文件而无法重新生成。为保护已有数据和结果，请使用独立工作副本。RapidOCR 脚本通过命令行接收图片路径，可使用独立工作副本。历史核验文件在 `evidence/`；其中 OCR 图片的原始本机路径已脱敏，保留了当时的相对位置，不能作为当前文件位置。

```bash
# 运行权威官方样本上的多框架横向评测
python scripts/benchmark_popular_frameworks.py

# 运行双栏阅读顺序穿插与基础模型评测
python scripts/benchmark_pdf_parsers.py
```

评测结果保存于 `results/popular_frameworks_comparison.json` 与 `results/pdf_parsers_benchmark.json`。

---

## 四、 OCR 引擎与 unstructured 元素切块评测

在前述规则解析器之外，补测两条端到端链路：图片 OCR 引擎（RapidOCR）与 unstructured 的
`partition_pdf` 元素切块。

1. **RapidOCR**（依赖 `rapidocr-onnxruntime`；模型和包版本未锁定）：
   - `synthetic-table.png`（合成表格图，取自 `table-structure-recognition` 实验 figures）
   - `testpage_cn.jpg`（中文扫描页，取自 `minicpm-ocr` 实验 evidence）
   - 指标：检出框数、耗时 ms（总时长及 det/cls/rec 分解，3 次取均值）、平均置信度；
2. **unstructured `partition_pdf`**：
   - `rapidocr_direct_and_image.pdf` 与 `pdfplumber_federal_register.pdf`；
   - 两条策略对比：`fast`（pdfminer 矢量流，纯规则、无模型）与
     `hi_res`（yolox ONNX 版面检测 + tesseract OCR + table-transformer 表格结构识别，
     `infer_table_structure=True`、`extract_images_in_pdf=True`）；
   - 指标：耗时、各类型元素数量（Title / NarrativeText / Table / Image 及完整直方图）；
   - 每个 (文件, 策略) 组合在独立子进程中冷启动计时。

历史环境为 Windows + Python 3.13；当时 `unstructured[pdf]` 的 extra 在该组合下不会带上
`unstructured-inference`，需手动补齐。记录的额外依赖有 `pi-heif`、`pikepdf`、`pdf2image`、
`unstructured-pytesseract`、`unstructured-inference`（含 torch CPU 版）；tesseract 二进制
当时用 scoop 安装并补 `eng.traineddata`。模型权重来自 Hugging Face；脚本检测到连接失败时会自动回退 `hf-mirror.com`，也可事先通过 `HF_ENDPOINT` 指定端点。OCR 和 unstructured 的完整版本与外部系统依赖尚未锁定，不能保证用当前版本一键重现历史结果。

可在独立虚拟环境安装 OCR 和 unstructured 路径的候选依赖；如只运行其中一条，只需安装对应清单：

```bash
python -m pip install -r requirements-rapidocr.txt -r requirements-unstructured.txt
python -m pip check
python -c "from rapidocr_onnxruntime import RapidOCR; from unstructured.partition.pdf import partition_pdf; print('Python 依赖可导入')"
tesseract --version
tesseract --list-langs
pdftoppm -v
```

按操作系统安装 Tesseract、英文 `eng` 语言数据和 Poppler 并加入 PATH；历史环境使用英文 OCR。包及模型版本未锁定，以上命令是准备起点，系统依赖与传递依赖仍需针对平台核对。RapidOCR 接收显式图片路径，只要有任一输入不存在就报错退出；成功后检查输出的 `images_evaluated`。unstructured 的 `hi_res` 还需获取模型权重。

```bash
# RapidOCR 图片 OCR 基准：替换为自行准备的图片路径
python scripts/benchmark_rapidocr.py /path/to/synthetic-table.png /path/to/testpage_cn.jpg

# unstructured partition_pdf 基准（fast vs hi_res）
python scripts/benchmark_unstructured.py

# 合并两部分结果并打印摘要
python scripts/merge_benchmark_results.py
```

合并后的完整评测数据保存于 `results/ocr_engines_and_unstructured_benchmark.json`
（分项数据在 `results/rapidocr_benchmark.json`、`results/unstructured_benchmark.json`）。历史记录见 [汇总证据](evidence/ocr_engines_and_unstructured_benchmark.json)。

2026-11-14 在 Linux ARM64、Python 3.12.3 的新独立环境中安装 `requirements-rapidocr.txt` 并通过 `pip check`；RapidOCR 1.4.4、ONNX Runtime 1.32.0 使用 CPU 执行提供方。两张仓库演示图片均完成预热和各 3 次计时运行：`testpage_cn.jpg` 检出 32 个框，`synthetic-table.png` 检出 12 个框。这验证了图片识别入口和该环境组合，不代表识别准确率、GPU 推理或 unstructured 已验证。

实测要点：hi_res 是唯一能产出 Table / Image 元素的路径（fast 按设计只走矢量文本流，
Table/Image 恒为 0）；15 页 Federal Register 上 hi_res 比 fast 慢约 24 倍
（113.2 s vs 4.7 s），检出 2 个表格、11 张图片，但产生大量 UncategorizedText；
fast 的 Title 计数（106）显著高于 hi_res（57），规则启发式对页眉页脚误判更多。
