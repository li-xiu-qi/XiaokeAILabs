# `sqlite-vec`

[![](https://dcbadge.vercel.app/api/server/VCtQ8cGhUs)](https://discord.gg/Ve7WeCJFXk)

一个极小、"足够快"的向量搜索 SQLite 扩展，可在任何地方运行！是 [`sqlite-vss`](https://github.com/asg017/sqlite-vss) 的接班人。

- 在 `vec0` 虚拟表中存储和查询浮点、int8 和二进制向量
- 使用纯 C 语言编写，无依赖，可在任何支持 SQLite 的地方运行
  （Linux/MacOS/Windows，浏览器 WASM，树莓派等）
- 在元数据、辅助或分区键列中存储非向量数据

## 安装

详见 [安装 `sqlite-vec`](https://alexgarcia.xyz/sqlite-vec/installation.html)。

| 语言          | 安装命令                                             | 更多信息                                                                             |                                                                                                                                                                                                    |
| -------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python         | `pip install sqlite-vec`                             | [`sqlite-vec` 与 Python](https://alexgarcia.xyz/sqlite-vec/python.html)             | [![PyPI](https://img.shields.io/pypi/v/sqlite-vec.svg?color=blue&logo=python&logoColor=white)](https://pypi.org/project/sqlite-vec/)                                                               |

## 使用示例

```sql
.load ./vec0

create virtual table vec_examples using vec0(
  sample_embedding float[8]
);

-- 向量可以用 JSON 或紧凑的二进制格式提供
insert into vec_examples(rowid, sample_embedding)
  values
    (1, '[-0.200, 0.250, 0.341, -0.211, 0.645, 0.935, -0.316, -0.924]'),
    (2, '[0.443, -0.501, 0.355, -0.771, 0.707, -0.708, -0.185, 0.362]'),
    (3, '[0.716, -0.927, 0.134, 0.052, -0.669, 0.793, -0.634, -0.162]'),
    (4, '[-0.710, 0.330, 0.656, 0.041, -0.990, 0.726, 0.385, -0.958]');


-- KNN 风格查询
select
  rowid,
  distance
from vec_examples
where sample_embedding match '[0.890, 0.544, 0.825, 0.961, 0.358, 0.0196, 0.521, 0.175]'
order by distance
limit 2;
/*
┌───────┬──────────────────┐
│ rowid │     distance     │
├───────┼──────────────────┤
│ 2     │ 2.38687372207642 │
│ 1     │ 2.38978505134583 │
└───────┴──────────────────┘
*/
```
