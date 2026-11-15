# Validating Jina CLIP Deployment on Intel CPU and NPU

This directory contains experiment code, configurations, and result evidence. See the [full experiment document](../../docs/zh/inference/OpenVINO与NPU多模态推理.md) for the purpose, principles, data, results, and boundaries.

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

Run all commands below from this experiment directory. See the [Chinese README](README.md).

## Reproduction

### 1. Environment and dependencies

Reproducing the device results requires Windows 11 x86-64, an Intel NPU, and the **32.0.100.5540 driver** used in this run; other versions need to be revalidated. The CPU check can run independently; the NPU check also requires OpenVINO to recognize the NPU. The requirements file pins OpenVINO, PyTorch, and Transformers, but not every other package, so it is not a complete environment lockfile. An older driver may reproduce historical failures, but those must not be mixed with current results. Official driver entry point: [Intel NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html).

Run the following PowerShell commands from this experiment directory. Set `$Work` to a work directory outside the repository and `$Image` to a local image you are authorized to use. `HF_HOME` redirects both model and custom-code caches. Reserve enough space for the model, IR, and results:

```powershell
$Work = 'C:\path\to\jina-work'
$Image = 'C:\path\to\your-image.jpg'
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$env:HF_HOME = "$Work\hf"
python -m venv "$Work\.venv"
$Py = "$Work\.venv\Scripts\python.exe"
& $Py -m pip install -r requirements-openvino.txt
& $Py -c "import openvino as ov; print(ov.Core().available_devices)"
```

The [jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) weights and custom implementation are downloaded at the fixed revision in the scripts and are not distributed with the repository. Prepare an external model cache and run-artifact directory, with enough space for downloads, conversion, and execution. The first run requires network access; later runs can use `--offline`. Downloads use the default Hugging Face Hub endpoint; if `HF_ENDPOINT` is already set, the scripts preserve it. The model card specifies [CC BY-NC 4.0](https://huggingface.co/jinaai/jina-clip-v2); confirm authorization separately before commercial use. Conversion uses `trust_remote_code=True`, so review the remote code at the pinned revision first.

### 2. CPU conversion and dual-branch validation

```powershell
& $Py scripts\run_openvino_smoke.py --device CPU --single-task --image "$Image" --work-dir "$Work\artifacts" --cache-dir "$Work\hf\hub"
```

This command creates static IR, a native reference, and a preprocessor, then runs a CPU comparison.

### 3. NPU dual-branch validation

After the model and custom implementation have been cached, use the same static inputs:

```powershell
& $Py scripts\run_openvino_smoke.py --device NPU --single-task --offline --image "$Image" --work-dir "$Work\artifacts" --cache-dir "$Work\hf\hub"
```

`$Image` must be an absolute path. The example no longer checks a specific label; because the image differs from the historical sample, this result is not a reproduction of the original visual result. Add `--force-convert` to rebuild the IR. When changing the image or text, use a new `--work-dir` to avoid reusing an old native reference.

### 4. Reuse existing IR for text comparison

```powershell
& $Py scripts\validate_existing_text_ir.py --device CPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\text-CPU.json"
& $Py scripts\validate_existing_text_ir.py --device NPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\text-NPU.json"
```

### 5. Repeated text benchmark

Run CPU first, then NPU, to avoid overlapping device tasks:

```powershell
& $Py scripts\benchmark_text_ir.py --device CPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\benchmark-CPU.json" --warmup 2 --iterations 10
& $Py scripts\benchmark_text_ir.py --device NPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\benchmark-NPU.json" --warmup 2 --iterations 10
```

### 6. Repeated vision benchmark

```powershell
& $Py scripts\benchmark_vision_ir.py --device CPU --ir "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --image "$Image" --output "$Work\artifacts\benchmark-vision-CPU.json" --warmup 2 --iterations 10
& $Py scripts\benchmark_vision_ir.py --device NPU --ir "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --image "$Image" --output "$Work\artifacts\benchmark-vision-NPU.json" --warmup 2 --iterations 10
```

### 7. Real image-text retrieval

Prepare 200 pairs on a machine that can access the Dataset Server and image source. This requires Python and `requests` (included in the Windows `requirements-openvino.txt`):

```powershell
& $Py scripts\prepare_coco_api_subset.py --output-dir "$Work\data\coco2017-val200" --limit 200
```

OpenVINO retrieval requires a document-side text IR without the task LoRA. First complete the CPU conversion above to prepare the local cache and preprocessor, then run this from the experiment directory on Windows:

```powershell
& $Py scripts\convert_document_text_ir.py --preprocessor "$Work\artifacts\fp16\preprocessor" --cache-dir "$Work\hf\hub" --out "$Work\artifacts\no-lora-fp16" --device CPU --offline
```

The conversion script sets the task LoRA to `None` by default, creates an IR from fixed inputs, and compares it on the selected device with native output for the same inputs. Use the IR only when `validation.json` has `ok` set to `true`. `--out` must name a new directory that does not exist. This entry point has not yet been rerun on the real model; if conversion or validation fails, do not use the IR or claim end-to-end reproduction. You can run `& $Py -m unittest discover -s tests -v`; mock-model and OpenVINO tests check that the task LoRA is disabled, the output-directory guard works, and validation failure exits. Tests do not load weights and cannot validate conversion or NPU inference. To validate another device, replace `--device NPU` and use a new output directory.

CUDA native runs in a Linux environment with an NVIDIA GPU. Create a separate environment as shown below, then install a CUDA build matching the GPU according to the [official PyTorch instructions](https://pytorch.org/get-started/locally/). The historical CUDA baseline used PyTorch 2.14.0+cu130 and Transformers 4.57.6; the new installation combination has not been verified from a clean setup.

Next, run OpenVINO NPU and CUDA native separately. The NPU command uses the Windows work directory above. On Linux, use a separate directory outside the repository and copy `data/coco2017-val200/` and `artifacts/fp16/preprocessor/` from the Windows work directory unchanged into the corresponding Linux work-directory paths. The first CUDA run needs network access to download the pinned model and custom code. If these files are fully prepared in the Linux `HF_HOME` cache, later runs can add `--offline`. When copying caches across devices, preserve cache links and custom code as well as model weights.

```powershell
& $Py scripts\evaluate_retrieval.py --backend openvino --device NPU --subset "$Work\data\coco2017-val200\subset.json" --query-ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --document-ir "$Work\artifacts\no-lora-fp16\jina-clip-text_v2_fp16.xml" --vision-ir "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\retrieval-coco200-NPU.json"
```

Run in the Linux experiment directory; replace `NATIVE_WORK` with the external work directory where files were copied:

```bash
export NATIVE_WORK=/path/to/jina-work
export HF_HOME="$NATIVE_WORK/hf"
mkdir -p "$NATIVE_WORK/artifacts"
python3 -m venv "$NATIVE_WORK/.venv"
source "$NATIVE_WORK/.venv/bin/activate"
# Install CUDA PyTorch appropriate for this device according to the official instructions first
python -m pip install -r requirements-native.txt
python -c "import torch; assert torch.cuda.is_available(), 'A CUDA GPU is required'"
python scripts/evaluate_retrieval.py --backend native --device cuda --subset "$NATIVE_WORK/data/coco2017-val200/subset.json" --preprocessor "$NATIVE_WORK/artifacts/fp16/preprocessor" --cache-dir "$NATIVE_WORK/hf/hub" --output "$NATIVE_WORK/artifacts/retrieval-coco200-native-cuda.json" --batch-size 8
```

### 8. Auxiliary checks

```powershell
& $Py scripts\compare_lora_modes.py --cache-dir "$Work\hf\hub" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\lora-disable-probe.json"
& $Py scripts\inspect_ir_dynamic_ops.py "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml"
& $Py scripts\inspect_ir_dynamic_ops.py "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml"
```
