# KNN查询 | sqlite-vec

向量在数据库中最常见的用例是K-近邻（KNN）查询。您会有一个向量表，并且想要找到K个最接近的

目前使用`sqlite-vec`执行KNN查询有两种方式：使用`vec0`虚拟表和使用常规表"手动"查询。

`vec0`虚拟表更快且更紧凑，但灵活性较低，需要`JOIN`回源表。"手动"方法更灵活且

## `vec0`虚拟表 ​

sql

```
create virtual table vec_documents using vec0(
  document_id integer primary key,
  contents_embedding float[768]
);

insert into vec_documents(document_id, contents_embedding)
  select id, embed(contents)
  from documents;
```

sql

```
select
  document_id,
  distance
from vec_documents
where contents_embedding match :query
  and k = 10;
```

sql

```
-- 此示例仅适用于SQLite 3.41+版本
-- 否则，请使用上面描述的`k = 10`方法！
select
  document_id,
  distance
from vec_documents
where contents_embedding match :query
limit 10; -- LIMIT仅在SQLite 3.41+版本有效
```

sql

```
with knn_matches as (
  select
    document_id,
    distance
  from vec_documents
  where contents_embedding match :query
    and k = 10
)
select
  documents.id,
  documents.contents,
  knn_matches.distance
from knn_matches
left join documents on documents.id = knn_matches.document_id
```

sql

```
create virtual table vec_documents using vec0(
  document_id integer primary key,
  contents_embedding float[768] distance_metric=cosine
);

-- 将向量插入到vec_documents...

-- 此MATCH现在将使用余弦距离而非默认的L2距离
select
  document_id,
  distance
from vec_documents
where contents_embedding match :query
  and k = 10;
```

## 使用SQL标量函数手动查询 ​

您不需要`vec0`虚拟表来用`sqlite-vec`执行KNN搜索。您可以在常规表的常规列中存储向量，如下所示：

sql

```
create table documents(
  id integer primary key,
  contents text,
  -- 一个4维浮点向量
  contents_embedding blob
);

insert into documents values
  (1, 'alex', vec_f32('[1.1, 1.1, 1.1, 1.1]')),
  (2, 'brian', vec_f32('[2.2, 2.2, 2.2, 2.2]')),
  (3, 'craig', vec_f32('[3.3, 3.3, 3.3, 3.3]'));
```

当您想找到相似向量时，可以手动使用`vec_distance_L2()`、`vec_distance_L1()`或`vec_distance_cosine()`，并通过`ORDER BY`子句执行暴力KNN查询。

sql

```
select
  id,
  contents,
  vec_distance_L2(contents_embedding, '[2.2, 2.2, 2.2, 2.2]') as distance
from documents
order by distance;

/*
┌────┬──────────┬──────────────────┐
│ id │ contents │     distance     │
├────┼──────────┼──────────────────┤
│ 2  │ 'brian'  │ 0.0              │
│ 3  │ 'craig'  │ 2.19999980926514 │
│ 1  │ 'alex'   │ 2.20000004768372 │
└────┴──────────┴──────────────────┘
*/
```

如果您选择这种方法，建议定义"向量列"时指定其元素类型（`float`、`bit`等）和维度，以便更好地文档化。还建议包含`CHECK`约束，以确保表中只存在正确元素类型和维度的向量。

sql

```
create table documents(
  id integer primary key,
  contents text,
  contents_embedding float[4]
    check(
      typeof(contents_embedding) == 'blob'
      and vec_length(contents_embedding) == 4
    )
);

-- ❌ 失败，需要BLOB输入
insert into documents values (1, 'alex', '[1.1, 1.1, 1.1, 1.1]');

-- ❌ 失败，3维，需要4维
insert into documents values (1, 'alex', vec_f32('[1.1, 1.1, 1.1]'));

-- ❌ 失败，需要float32向量
insert into documents values (1, 'alex', vec_bit('[1.1, 1.1, 1.1, 1.1]'));

-- ✅ 成功！
insert into documents values (1, 'alex', vec_f32('[1.1, 1.1, 1.1, 1.1]'));
```

请记住：**SQLite不支持自定义类型**。上面的示例可能看起来像`contents_embedding`列有一个`float[4]`的"自定义类型"，但SQLite允许_任何内容_作为"列类型"。

sql

```
-- 这些"列类型"在SQLite中完全合法
create table students(
  name ham_sandwich,
  age minions[42]
);
```

更多信息请参阅SQLite中的数据类型。

因此，`float[4]`作为"列类型"本身完全不受SQLite强制执行。这就是为什么我们建议包含`CHECK`约束，以确保向量列中的值具有正确的类型和长度。

对于严格表，使用`BLOB`类型并包含相同的`CHECK`约束。

sql

```
create table documents(
  id integer primary key,
  contents text,
  contents_embedding blob check(vec_length(contents_embedding) == 4)
) strict;
```
