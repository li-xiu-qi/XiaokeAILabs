# 前端设计

## 1. 技术栈

- **框架**：React
- **UI 组件库**：Ant Design
- **HTTP 请求**：Axios/Fetch API
- **状态管理**：React 状态钩子 (useState, useEffect) 或 Redux Toolkit/Zustand

## 2. 页面结构

### 2.1. 主页面

主页面采用垂直分区布局，从上到下包括：

- 顶部导航栏：系统名称、版本信息、可能的操作按钮
- 输入区域：文件上传和文本输入区
- 操作区域：生成按钮和其他控制功能
- QA 对展示与审核区域：表格或列表形式
- 底部操作区：提交审核、导出 JSON 按钮等

### 2.2. 布局设计

```
+------------------------------------------+
|             导航栏/标题栏                |
+------------------------------------------+
|                                          |
|              输入区域                    |
|   +-------------+    +-------------+     |
|   | 文件上传区   |    | 文本输入区   |     |
|   +-------------+    +-------------+     |
|                                          |
+------------------------------------------+
|                                          |
|              操作按钮                    |
|        [生成 QA 对]  [清空]              |
|                                          |
+------------------------------------------+
|                                          |
|           QA 对展示与审核区              |
|   +-----------------------------------+  |
|   | 问题 | 答案 | 评分 | 操作        |  |
|   |------|------|------|------------|  |
|   |  Q1  |  A1  | 评分1 | 编辑/删除  |  |
|   |  Q2  |  A2  | 评分2 | 编辑/删除  |  |
|   |  ... |  ... | ...  | ...        |  |
|   +-----------------------------------+  |
|                                          |
+------------------------------------------+
|                                          |
|              底部操作区                  |
|        [提交审核]    [导出 JSON]         |
|                                          |
+------------------------------------------+
```

## 3. 主要组件设计

### 3.1. 输入组件

#### 3.1.1. 文件上传组件

使用 Ant Design 的 `Upload` 组件实现文件上传功能：

```jsx
<Upload.Dragger
  name="file"
  accept=".txt,.md,.pdf,.docx"
  beforeUpload={(file) => {
    // 验证文件类型
    return false; // 阻止自动上传
  }}
  onChange={(info) => {
    // 处理文件变更
    setFile(info.file);
  }}
>
  <p className="ant-upload-drag-icon">
    <InboxOutlined />
  </p>
  <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
  <p className="ant-upload-hint">
    支持 .txt、.md、.pdf 和 .docx 格式的文件
  </p>
</Upload.Dragger>
```

#### 3.1.2. 文本输入组件

使用 Ant Design 的 `Input.TextArea` 组件实现文本输入功能：

```jsx
<Input.TextArea
  placeholder="在此输入文本..."
  autoSize={{ minRows: 4, maxRows: 12 }}
  value={text}
  onChange={(e) => setText(e.target.value)}
/>
```

### 3.2. QA 对展示与审核组件

使用 Ant Design 的 `Table` 组件显示 QA 对和评分信息：

```jsx
<Table
  dataSource={qaPairs}
  rowKey={(record) => record.id}
  columns={[
    {
      title: '问题',
      dataIndex: 'question',
      key: 'question',
      render: (text, record) => (
        <div>
          {editingId === record.id ? (
            <Input.TextArea
              value={editingQuestion}
              onChange={(e) => setEditingQuestion(e.target.value)}
              autoSize={{ minRows: 2, maxRows: 6 }}
            />
          ) : (
            text
          )}
        </div>
      ),
    },
    {
      title: '答案',
      dataIndex: 'answer',
      key: 'answer',
      render: (text, record) => (
        <div>
          {editingId === record.id ? (
            <Input.TextArea
              value={editingAnswer}
              onChange={(e) => setEditingAnswer(e.target.value)}
              autoSize={{ minRows: 2, maxRows: 6 }}
            />
          ) : (
            text
          )}
        </div>
      ),
    },
    {
      title: '评分',
      dataIndex: 'scores',
      key: 'scores',
      render: (scores) => (
        <Space direction="vertical">
          {Object.entries(scores).map(([key, value]) => (
            <Tag color={getScoreColor(value)} key={key}>
              {key}: {value}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="middle">
          {editingId === record.id ? (
            <>
              <Button type="primary" onClick={() => saveEdit(record.id)}>
                保存
              </Button>
              <Button onClick={() => cancelEdit()}>取消</Button>
            </>
          ) : (
            <>
              <Button type="text" icon={<EditOutlined />} onClick={() => startEdit(record)}>
                编辑
              </Button>
              <Button type="text" danger icon={<DeleteOutlined />} onClick={() => deletePair(record.id)}>
                删除
              </Button>
            </>
          )}
        </Space>
      ),
    },
  ]}
/>
```

### 3.3. 操作按钮组件

使用 Ant Design 的 `Button` 组件实现操作按钮：

```jsx
<div className="button-group">
  <Button 
    type="primary" 
    icon={<PlayCircleOutlined />} 
    onClick={handleGenerate}
    loading={isGenerating}
    disabled={!file && !text}
  >
    生成 QA 对
  </Button>
  <Button 
    icon={<ClearOutlined />} 
    onClick={handleClear}
  >
    清空
  </Button>
</div>

<div className="bottom-actions">
  <Button 
    type="primary" 
    icon={<CheckOutlined />} 
    onClick={handleSubmitReview}
    disabled={qaPairs.length === 0}
  >
    提交审核
  </Button>
  <Button 
    icon={<DownloadOutlined />} 
    onClick={handleExportJSON}
    disabled={!canExport}
  >
    导出 JSON
  </Button>
</div>
```

## 4. 状态管理

### 4.1. 主要状态

```jsx
// 输入状态
const [file, setFile] = useState(null);
const [text, setText] = useState('');

// QA 对状态
const [qaPairs, setQaPairs] = useState([]);
const [reviewedPairs, setReviewedPairs] = useState([]);

// 编辑状态
const [editingId, setEditingId] = useState(null);
const [editingQuestion, setEditingQuestion] = useState('');
const [editingAnswer, setEditingAnswer] = useState('');

// UI 状态
const [isGenerating, setIsGenerating] = useState(false);
const [isSubmitting, setIsSubmitting] = useState(false);
const [isExporting, setIsExporting] = useState(false);
const [error, setError] = useState(null);
const [canExport, setCanExport] = useState(false);
```

### 4.2. 副作用管理

```jsx
// 当 reviewedPairs 变更时，更新导出状态
useEffect(() => {
  setCanExport(reviewedPairs.length > 0);
}, [reviewedPairs]);

// 处理错误显示
useEffect(() => {
  if (error) {
    message.error(error);
    setError(null);
  }
}, [error]);
```

## 5. API 交互

### 5.1. 文件上传与 QA 生成

```jsx
const handleGenerate = async () => {
  setIsGenerating(true);
  try {
    let response;
    
    if (file) {
      // 文件上传处理
      const formData = new FormData();
      formData.append('file', file);
      
      response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
    } else if (text) {
      // 文本输入处理
      response = await axios.post('/api/generate_from_text', { text });
    }
    
    if (response && response.data) {
      setQaPairs(response.data.qa_pairs);
    }
  } catch (err) {
    setError('生成 QA 对失败：' + (err.response?.data?.detail || err.message));
  } finally {
    setIsGenerating(false);
  }
};
```

### 5.2. 提交审核

```jsx
const handleSubmitReview = async () => {
  setIsSubmitting(true);
  try {
    const response = await axios.post('/api/submit_review', { qa_pairs: qaPairs });
    if (response && response.data) {
      setReviewedPairs(response.data.qa_pairs);
      message.success('审核提交成功！');
    }
  } catch (err) {
    setError('提交审核失败：' + (err.response?.data?.detail || err.message));
  } finally {
    setIsSubmitting(false);
  }
};
```

### 5.3. 导出 JSON

```jsx
const handleExportJSON = async () => {
  setIsExporting(true);
  try {
    const response = await axios.get('/api/export_json', { responseType: 'blob' });
    
    // 创建下载链接
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `qa_pairs_${new Date().getTime()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    message.success('JSON 导出成功！');
  } catch (err) {
    setError('导出 JSON 失败：' + err.message);
  } finally {
    setIsExporting(false);
  }
};
```

## 6. UI/UX 注意事项

### 6.1. 响应式设计

使用 Ant Design 的 Grid 系统实现响应式布局：

```jsx
<Row gutter={[16, 16]}>
  <Col xs={24} md={12}>
    {/* 文件上传区域 */}
  </Col>
  <Col xs={24} md={12}>
    {/* 文本输入区域 */}
  </Col>
</Row>
```

### 6.2. 加载状态反馈

使用 Ant Design 的 `Spin` 组件和按钮的 `loading` 属性提供视觉反馈：

```jsx
<Spin spinning={isGenerating || isSubmitting || isExporting} tip="处理中...">
  {/* 页面内容 */}
</Spin>
```

### 6.3. 错误处理与提示

使用 Ant Design 的 `message` 和 `notification` 组件显示操作结果：

```jsx
// 成功提示
message.success('操作成功！');

// 错误提示
message.error('操作失败：' + errorMessage);

// 重要通知
notification.info({
  message: '提示',
  description: '请先上传文件或输入文本',
  placement: 'topRight'
});
```