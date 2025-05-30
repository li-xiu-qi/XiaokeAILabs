### SQLite 数据类型

SQLite 采用动态类型系统，这意味着数据类型与值本身相关联，而不是与存储值的容器（列）相关联。尽管如此，为了与其他 SQL 数据库引擎兼容，SQLite 使用“类型亲和性”的概念。类型亲和性是指为列中存储的数据推荐的类型。

**SQLite 主要有五种存储类别：**

* **NULL**：表示空值。
* **INTEGER**：表示整数值。布尔值也存储为整数（0 表示 false，1 表示 true）。
* **REAL**：表示浮点数值。
* **TEXT**：表示文本字符串。
* **BLOB** (Binary Large Object)：用于存储二进制大对象，例如图像或文件，按输入原样存储。

**类型亲和性：**

SQLite 中的每一列都会被分配以下五种类型亲和性之一：

* **TEXT**：具有 TEXT 亲和性的列会存储所有数据为 NULL、TEXT 或 BLOB 类型。
* **NUMERIC**：具有 NUMERIC 亲和性的列可以存储所有五种存储类别的数据。在可能的情况下，它会将文本转换为 INTEGER 或 REAL。
* **INTEGER**：行为与 NUMERIC 类似。
* **REAL**：具有 REAL 亲和性的列会将整数值强制转换为浮点表示。
* **BLOB**：具有 BLOB 亲和性的列不偏好任何存储类别，也不会尝试强制转换数据。

**日期和时间：**

SQLite 没有单独的日期和时间存储类别。相反，它使用内置函数将日期和时间存储为以下三种格式之一：

* **TEXT**：ISO8601 格式的字符串。
* **REAL**：儒略日数。
* **INTEGER**：Unix 时间戳。

在查询执行期间，SQLite 可能会在数字存储类别（INTEGER 和 REAL）与 TEXT 之间转换值。
