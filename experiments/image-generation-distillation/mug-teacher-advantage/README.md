# 陶瓷杯图像：教师方案的开发集检查

[English](README.en.md) | [实验入口](../README.md)

在 32 条配对开发提示上，SDXL 教师方案以 1024 像素生成再缩小至 512 像素；SD 1.5 基线直接生成 512 像素。两者均使用 30 步、guidance 7.5 和各自原生调度器，两位 AI 评审独立盲评。

预先指定的第一位评审：32 对有效，教师复合成功 23 次、基线 7 次，教师独赢 18 对、基线独赢 2 对。第二位评审：29 对有效、3 对未知，教师成功 22 次、基线 7 次。两位均满足原定门槛：有效对数至少 28、教师成功至少 20、教师比基线至少多 4 次。

排除任一方填 unknown 的项目，两人的逐图一致数为：单杯 61/62、杯身颜色 63/64、背景颜色 61/64、严重伪影 61/62。单杯、颜色和背景均正确且无严重伪影才算复合成功；任一项 unknown 的配对不计有效。

该结果支持在此场景继续训练对照，并不证明蒸馏成功；模型、分辨率及调度器共同变化，无法单独归因。评审均为 AI；样本量小；本地冻结记录没有第三方可信时间戳。仓库目录保留精简评分材料；[开发集证据附件](https://github.com/li-xiu-qi/XiaokeAILabs/releases/tag/evidence-mug-development-20261107)另行提供原图、评分图、评分与哈希，不包含模型权重。这是开发集检查，不是最终测试。

## 重算评分

Python 3.10 及以上，标准库即可：

```bash
python reveal_compact.py --review review-1.csv --key mug-key-1.json --out replay-1.csv
python reveal_compact.py --review review-2.csv --key mug-key-2.json --out replay-2.csv
python check_mug_gate.py --scores replay-1.csv --development development.jsonl
python check_mug_gate.py --scores replay-2.csv --development development.jsonl
```

脚本检查匿名编号、提示和评分，重建配对并计算门槛，不验证图片本身。

