# sqlite-vss

> [!警告]  
> `sqlite-vss` 目前不再积极开发。取而代之的是我现在致力于开发 [`sqlite-vec`](https://github.com/asg017/sqlite-vec)，这是一个类似的向量搜索SQLite扩展，但比`sqlite-vss`更易于安装和使用。更多信息请参阅[这篇博客文章](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html)。
>

`sqlite-vss`（SQLite <b><u>V</u></b>ector <b><u>S</u></b>imilarity <b><u>S</u></b>earch，向量相似度搜索）是一个基于[Faiss](https://faiss.ai/)的SQLite扩展，为SQLite带来向量搜索功能。它可以用于构建语义搜索引擎、推荐系统或问答工具。

查看[_Introducing sqlite-vss: A SQLite Extension for Vector Search_](https://observablehq.com/@asg017/introducing-sqlite-vss)（2023年2月）获取更多详情和实时示例！

如果您的公司或组织发现此库有用，请考虑[支持我的工作](#supporting)！

## 使用方法

```sql
.load ./vector0
.load ./vss0

select vss_version(); -- 'v0.0.1'

```

`sqlite-vss`的API与[`fts5`全文搜索扩展](https://www.sqlite.org/fts5.html)类似。使用`vss0`模块创建虚拟表，可以高效存储和查询您的向量。

```sql
-- 384 == 此示例的维度数量
create virtual table vss_articles using vss0(
  headline_embedding(384),
  description_embedding(384),
);
```

`sqlite-vss`是一个**自带向量**的数据库，它兼容任何您拥有的嵌入或向量数据。考虑使用[OpenAI的嵌入API](https://platform.openai.com/docs/guides/embeddings)、[HuggingFace的推理API](https://huggingface.co/blog/getting-started-with-embeddings#1-embedding-a-dataset)、[`sentence-transformers`](https://www.sbert.net/)或[任何这些开源模型](https://github.com/embeddings-benchmark/mteb#leaderboard)。在此示例中，我们使用[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)从文本生成384维的嵌入向量。

您可以以JSON或原始字节的形式将向量插入到`vss0`表中。

```sql
insert into vss_articles(rowid, headline_embedding)
  select rowid, headline_embedding from articles;
```

要查询相似向量（"k近邻"），请在`WHERE`子句中使用`vss_search`函数。这里我们搜索`articles`表中行#123的嵌入向量的100个最近邻。

```sql
select rowid, distance
from vss_articles
where vss_search(
  headline_embedding,
  (select headline_embedding from articles where rowid = 123)
)
limit 100;
```

您可以根据需要对这些表进行`INSERT`和`DELETE`操作，但[`UPDATE`操作尚不支持](https://github.com/asg017/sqlite-vss/issues/7)。这可以与触发器一起使用，自动更新索引。另请注意，仅插入几行的"小型"`INSERT`/`DELETE`操作可能会很慢，因此请尽可能批量操作。

```sql
begin;

delete from vss_articles
  where rowid between 100 and 200;

insert into vss_articles(rowid, headline_embedding, description_embedding)
  values (:rowid, :headline_embedding, :description_embedding)

commit;
```

您可以为特定列传入自定义[Faiss工厂字符串](https://github.com/facebookresearch/faiss/wiki/The-index-factory)来控制Faiss索引的存储和查询方式。默认情况下，工厂字符串是`"Flat,IDMap2"`，随着数据库增长，查询可能变得缓慢。在这里，我们添加了带有4096个质心的[倒排文件索引](https://github.com/facebookresearch/faiss/wiki/The-index-factory#inverted-file-indexes)，这是一个非穷举选项，能使大型数据库查询速度更快。

```sql
create virtual table vss_ivf_articles using vss0(
  headline_embedding(384) factory="IVF4096,Flat,IDMap2",
  description_embedding(384) factory="IVF4096,Flat,IDMap2"
);
```

这个IVF需要训练！您可以通过在单个事务中执行`INSERT`命令并使用特殊的`operation="training"`约束来定义训练数据。

```sql
insert into vss_ivf_articles(operation, headline_embedding, description_embedding)
  select
    'training',
    headline_embedding,
    description_embedding
  from articles;
```

请注意！需要训练的索引可能需要很长时间。在这个例子中使用的[新闻类别数据集](./examples/headlines/)（210k个向量，每个386维）中，默认索引构建需要8秒。但使用自定义的`"IVF4096,Flat,IDMap2"`工厂，训练需要45分钟，插入数据需要4.5分钟！这可能可以通过使用较小的训练集来减少，但更快的查询是有益的。

## 文档

查看[`docs.md`](./docs.md)了解自行编译`sqlite-vss`的说明，以及完整的SQL API参考。

## 安装

[Releases页面](https://github.com/asg017/sqlite-vss/releases)包含适用于Linux x86_64和MacOS x86_64（MacOS Big Sur 11或更高版本）的预构建二进制文件。未来将提供更多预编译目标。此外，`sqlite-vss`还分发在常见的包管理器上，如Python的`pip`和Node.js的`npm`，详情见下文。

请注意，在Linux机器上，您需要安装一些软件包才能使这些选项工作：

```
sudo apt-get update
sudo apt-get install -y libgomp1 libatlas-base-dev liblapack-dev
```

> **注意：**
> 文件名中的`0`（`vss0.dylib`/`vss0.so`）表示`sqlite-vss`的主要版本。目前`sqlite-vss`处于v1之前的阶段，因此未来版本可能会有重大变更。

| 语言           | 安装                                                             | 更多信息                                                             |                                                                                                                                                                                           |
| -------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python         | `pip install sqlite-vss`                                          | [`sqlite-vss`与Python配合使用](https://alexgarcia.xyz/sqlite-vss/python.html)       | [![PyPI](https://img.shields.io/pypi/v/sqlite-vss.svg?color=blue&logo=python&logoColor=white)](https://pypi.org/project/sqlite-vss/)                                                      |

### 与`sqlite3`命令行界面一起使用

要在[官方SQLite命令行shell](https://www.sqlite.org/cli.html)中使用`sqlite-vss`，请从发布版本中下载`vector0.dylib`/`vss0.dylib`（适用于MacOS Big Sur 11或更高版本）或`vector0.so`/`vss0.so`（Linux）文件，并将其加载到您的SQLite环境中。

`vector0`扩展是必需的依赖项，因此请确保在加载`vss0`之前先加载它。

```sql
.load ./vector0
.load ./vss0
select vss_version();
-- v0.0.1
```

### Python

对于Python开发者，通过以下命令安装[`sqlite-vss`包](https://pypi.org/package/sqlite-vss/)：

```
pip install sqlite-vss
```

```python
import sqlite3
import sqlite_vss

db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vss.load(db)

version, = db.execute('select vss_version()').fetchone()
print(version)
```

有关更多详细信息，请参阅[`bindings/python`](./bindings/python/README.md)。

## 缺点

- 底层Faiss索引的大小上限为1GB。关注[#1](https://github.com/asg017/sqlite-vss/issues/1)获取更新。
- 目前不支持在KNN搜索之上进行额外的过滤。关注[#2](https://github.com/asg017/sqlite-vss/issues/2)获取更新。
- 仅支持CPU Faiss索引，尚不支持GPU。关注[#3](https://github.com/asg017/sqlite-vss/issues/3)获取更新。
- 尚不支持内存映射(mmap)索引，因此索引必须适合RAM。关注[#4](https://github.com/asg017/sqlite-vss/issues/4)获取更新。
- 此扩展使用C++编写，尚无模糊测试。关注[#5](https://github.com/asg017/sqlite-vss/issues/5)获取更新。
- vss0虚拟表不支持`UPDATE`语句，但支持`INSERT`和`DELETE`语句。关注[#7](https://github.com/asg017/sqlite-vss/issues/7)获取更新。
