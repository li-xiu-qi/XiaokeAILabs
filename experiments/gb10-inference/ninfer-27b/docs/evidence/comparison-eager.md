# DGX Spark：FP8 与 NVFP4 W4A4 + DFlash2 对照结果

两档均使用 SGLang、DFlash2 草稿 8、512 token 上限与 xhigh 思考。吞吐是客户端端到端口径，包含 prefill 与排队，不等于纯 decode tok/s。小样本质量探针不能替代完整公开测评。

## 并发性能

| 并发 | NVFP4 总输出 tok/s | FP8 总输出 tok/s | FP8/NVFP4 | NVFP4 单路中位 tok/s | FP8 单路中位 tok/s | NVFP4 接受率 | FP8 接受率 | NVFP4 接受长度 | FP8 接受长度 | 完成（NV/FP） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.7 | 22.9 | 0.70x | 32.8 | 22.9 | 35.7% | 36.0% | 3.51 | 3.51 | 0/2 / 0/2 |
| 2 | 55.2 | 45.7 | 0.83x | 28.7 | 24.6 | 32.4% | 43.7% | 3.26 | 4.06 | 0/4 / 1/4 |
| 4 | 96.9 | 68.5 | 0.71x | 31.1 | 22.7 | 40.1% | 40.8% | 3.80 | 3.85 | 2/8 / 2/8 |
| 8 | 132.5 | 106.1 | 0.80x | 21.9 | 17.2 | 32.9% | 34.6% | 3.30 | 3.42 | 1/16 / 2/16 |

## 小样本质量探针

自动题共 10 道；编程题只检查语法和函数名，不计入自动正确率。截断与空答单独列出。

| 档位 | 自动题通过 | 截断 | 空答 | 编程语法通过 |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 W4A4 | 10/10 | 0 | 0 | 2/2 |
| FP8 | 10/10 | 0 | 0 | 2/2 |

## 文本与视觉冒烟

| 档位 | 总体通过 | 检查项 |
| --- | --- | --- |
| NVFP4 W4A4 | True | text_has_391=True, text_finished=True, image_finished=True, image_has_red_circle=True, image_has_blue_square=True, image_has_text=True |
| FP8 | True | text_has_391=True, text_finished=True, image_finished=True, image_has_red_circle=True, image_has_blue_square=True, image_has_text=True |
