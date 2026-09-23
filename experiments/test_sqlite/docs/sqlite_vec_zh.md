# 我正在开发一个新的向量搜索SQLite扩展

2024-05-02 作者：Alex Garcia

> _简而言之 — `sqlite-vec`将是一个新的_
> _SQLite向量搜索扩展，取代`sqlite-vss`。它将是一个_
> _可嵌入的"足够快"的向量搜索工具，可在任何支持SQLite的地方运行 - 包括WASM！它仍在积极开发中，但_

* * *

我正在开发一个新的SQLite扩展！它叫做`sqlite-vec`，一个完全用C语言编写的向量搜索扩展。它旨在取代我在2023年2月发布的另一个向量搜索SQLite扩展`sqlite-vss`，后者存在一些问题。我相信`sqlite-vec`采用的方法解决了前任的许多问题，将拥有更好、性能更高的SQL API，并且更适合所有需要嵌入式向量搜索解决方案的应用！

## ¶ `sqlite-vec`将会是什么

`sqlite-vec`将是一个纯C语言编写、无依赖的SQLite扩展。它将提供自定义SQL函数和虚拟表用于快速向量搜索，以及其他处理向量的工具和实用程序（量化、JSON/BLOB/numpy转换、向量运算等）。

下面是使用`sqlite-vec`进行向量搜索的一个简单示例，全部用纯SQL：

```sql
.load ./vec0

-- 用于8维浮点数的"向量存储"
create virtual table vec_examples using vec0(
  sample_embedding float[8]
);

-- 向量可以以JSON格式或紧凑的二进制格式提供
insert into vec_examples(rowid, sample_embedding)
  values
    (1, '[-0.200, 0.250, 0.341, -0.211, 0.645, 0.935, -0.316, -0.924]'),
    (2, X'E5D0E23E894100BF8FC2B53E426045BFF4FD343F7D3F35BFA4703DBE1058B93E'),
    (3, '[0.716, -0.927, 0.134, 0.052, -0.669, 0.793, -0.634, -0.162]'),
    (4, X'8FC235BFC3F5A83E9EEF273F9EEF273DA4707DBF23DB393FB81EC53E7D3F75BF');

-- KNN风格查询飞速运行
  select
    rowid,
    distance
  from vec_examples
  where sample_embedding match '[0.890, 0.544, 0.825, 0.961, 0.358, 0.0196, 0.521, 0.175]'
  order by distance
  limit 2;

/*
rowid,distance
2,2.38687372207642
1,2.38978505134583
*/
```

使用`sqlite-vec`意味着使用纯SQL，只需`CREATE VIRTUAL TABLE`、`INSERT INTO`和`SELECT`语句。

这项工作令人兴奋，有很多原因！首先，"纯C语言编写"意味着它可以在任何地方运行。之前的`sqlite-vss`扩展因有一些繁琐的C++依赖项，只能可靠地在Linux和MacOS机器上运行，二进制文件大小在3MB-5MB范围内。相比之下，`sqlite-vec`将在所有平台（Linux/MacOS/Windows）上运行，在浏览器中使用WebAssembly，甚至在手机和树莓派等更小的设备上也能运行！二进制文件也更小，仅几百KB大小。

此外，`sqlite-vec`对内存使用有更多控制。默认情况下，向量存储在影子表的"块"中，并在KNN搜索期间逐块读取。这意味着你不需要将所有内容都存储在RAM中！不过如果你想要内存级速度，可以使用`PRAGMA mmap_size`命令使KNN搜索更快。

最后，`sqlite-vec`是在向量搜索工具和研究的新"时代"中构建的。它将更好地支持"自适应长度嵌入"（又称俄罗斯套娃嵌入），以及`int8`/`bit`向量支持二进制和标量量化。这意味着对向量占用的速度、精度和磁盘空间有更多控制。

不过最初，`sqlite-vec`将只支持穷举式全扫描向量搜索。不会有"近似最近邻"（ANN）选项。但我希望将来添加IVF + HNSW！

## 但`sqlite-vss`有什么问题？

我不会详细说明所有细节，但`sqlite-vss`的开发和采用存在许多障碍，包括：

- 只能在Linux + MacOS机器上工作（不支持Windows、WASM、移动设备等）
- 将所有向量都存储在内存中
- 各种与事务相关的bug和问题
- 编译极其困难且耗时
- 缺少常见的向量操作（标量/二进制量化）

几乎所有这些问题都是因为`sqlite-vss`依赖于Faiss。花费大量时间和精力，其中一些问题也许可以解决，但许多问题会被Faiss阻碍。

考虑到这一点，一个无依赖和低级别的解决方案看起来非常诱人。事实证明，向量搜索并不太复杂，所以`sqlite-vec`诞生了！
