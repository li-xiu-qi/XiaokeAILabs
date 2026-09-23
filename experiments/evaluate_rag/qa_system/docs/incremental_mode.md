# 增量处理模式

## 1. 概述

增量处理模式允许用户通过多次操作来逐步构建和扩充QA对数据集，而不是一次性处理完所有文本并导出结果。这种模式特别适合以下场景：

- 用户有多个文档需要依次处理
- 文档内容经常更新，需要定期追加新的QA对
- 用户希望对不同来源的文本生成的QA对进行分批审核和管理
- 项目团队希望协作构建大型QA知识库

## 2. 核心功能

### 2.1. 状态保持与累积

系统将维护当前会话中所有已生成并审核的QA对，新生成的QA对将添加到现有集合中。具体功能包括：

- 会话持久化：在前后端都保持当前会话的QA对集合
- 去重机制：检测重复或高度相似的问题，避免重复添加
- 分批查看：支持按来源文档、生成批次等方式筛选和查看QA对

### 2.2. 多次文件上传

系统将支持多次上传文件，并且每次处理后都可以累积QA对结果：

- 支持单次或多选上传markdown文件
- 文件处理历史记录，显示已处理过的文件
- 每个QA对关联到其源文件信息

### 2.3. 分批文本处理

除了文件上传，还支持用户多次输入文本片段：

- 文本输入历史记录，记录已处理的文本片段
- 提供文本片段分组功能，便于后期管理
- 支持从剪贴板直接粘贴文本并处理

## 3. 界面设计

### 3.1. QA对集合管理面板

在UI界面中添加一个QA对管理面板，包含以下功能：

```
+------------------------------------------+
|            QA 对集合管理面板              |
+------------------------------------------+
| 当前数量: 35 个QA对                       |
| 批次数量: 5 批次                          |
|                                          |
| [ 查看全部 ] [ 按批次筛选 ▼ ] [ 按来源筛选 ▼ ]  |
|                                          |
| [批量操作 ▼]   [清空集合]   [导出当前集合]  |
+------------------------------------------+
```

### 3.2. 处理历史记录

添加处理历史记录标签页，记录文件上传和文本处理的历史：

```
+------------------------------------------+
|              处理历史                    |
+------------------------------------------+
| 来源     | 类型   | 时间        | QA对数量 |
|---------|-------|------------|---------|
| file1.md| 文件   | 2025-04-24 | 12      |
| 手动输入 | 文本   | 2025-04-24 | 5       |
| file2.md| 文件   | 2025-04-25 | 18      |
| ...     | ...   | ...        | ...     |
+------------------------------------------+
```

### 3.3. 增量上传组件

修改文件上传组件以支持增量模式：

```jsx
<Upload.Dragger
  name="file"
  accept=".md"  // 只接受markdown文件
  multiple={true}  // 支持多文件上传
  fileList={fileList}
  onChange={handleFileChange}
  beforeUpload={(file) => {
    // 验证文件类型
    const isMarkdown = file.type === 'text/markdown' || file.name.endsWith('.md');
    if (!isMarkdown) {
      message.error('只能上传 Markdown 文件！');
      return false;
    }
    // 阻止自动上传
    return false;
  }}
>
  <p className="ant-upload-drag-icon">
    <InboxOutlined />
  </p>
  <p className="ant-upload-text">点击或拖拽 Markdown 文件到此区域上传</p>
  <p className="ant-upload-hint">
    支持单个或批量上传，QA对将累积到当前集合中
  </p>
</Upload.Dragger>
```

## 4. 后端实现

### 4.1. 会话管理

后端需要维护用户的会话状态，存储已生成和审核的QA对：

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
import uuid
from typing import Dict, List
from datetime import datetime

# 简单的会话存储（生产环境应使用数据库）
sessions: Dict[str, dict] = {}

def create_session():
    """创建新的会话"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "qa_pairs": [],
        "history": [],
        "created_at": datetime.now(),
        "last_active": datetime.now()
    }
    return session_id

def get_session(session_id: str = Depends(APIKeyHeader(name="X-Session-ID", auto_error=False))):
    """获取会话数据"""
    if not session_id or session_id not in sessions:
        new_id = create_session()
        return new_id, sessions[new_id]
    
    # 更新最后活动时间
    sessions[session_id]["last_active"] = datetime.now()
    return session_id, sessions[session_id]

# 会话清理任务（可配置为定时执行）
def cleanup_old_sessions(max_age_hours=24):
    """清理超过指定时间未活动的会话"""
    now = datetime.now()
    to_delete = []
    for session_id, data in sessions.items():
        if (now - data["last_active"]).total_seconds() > max_age_hours * 3600:
            to_delete.append(session_id)
    
    for session_id in to_delete:
        del sessions[session_id]
```

### 4.2. API 端点

添加新的API端点支持增量模式操作：

```python
@router.post("/upload_incremental", response_model=QAResponse)
async def upload_incremental(
    files: List[UploadFile] = File(...),
    session_data: Tuple[str, dict] = Depends(get_session)
):
    """增量处理上传的文件"""
    session_id, session = session_data
    all_qa_pairs = []
    
    for file in files:
        # 验证文件类型
        if not file.filename.endswith('.md'):
            continue
            
        # 处理文件
        text = await parse_file_to_text(file)
        new_qa_pairs = await generate_qa_from_text(text)
        
        # 记录来源
        for qa in new_qa_pairs:
            qa.source = file.filename
            qa.batch_id = str(uuid.uuid4())
        
        # 添加到会话
        session["qa_pairs"].extend(new_qa_pairs)
        
        # 添加到历史记录
        session["history"].append({
            "source": file.filename,
            "type": "file",
            "timestamp": datetime.now().isoformat(),
            "count": len(new_qa_pairs)
        })
        
        all_qa_pairs.extend(new_qa_pairs)
    
    return {
        "session_id": session_id,
        "qa_pairs": all_qa_pairs,
        "total_count": len(session["qa_pairs"])
    }

@router.post("/text_incremental", response_model=QAResponse)
async def process_text_incremental(
    request: TextGenerationRequest,
    session_data: Tuple[str, dict] = Depends(get_session)
):
    """增量处理文本输入"""
    session_id, session = session_data
    
    # 生成 QA 对
    new_qa_pairs = await generate_qa_from_text(request.text)
    
    # 添加批次和来源信息
    batch_id = str(uuid.uuid4())
    for qa in new_qa_pairs:
        qa.source = "manual_input"
        qa.batch_id = batch_id
    
    # 添加到会话
    session["qa_pairs"].extend(new_qa_pairs)
    
    # 添加到历史记录
    session["history"].append({
        "source": "manual_input",
        "type": "text",
        "timestamp": datetime.now().isoformat(),
        "count": len(new_qa_pairs)
    })
    
    return {
        "session_id": session_id,
        "qa_pairs": new_qa_pairs,
        "total_count": len(session["qa_pairs"])
    }

@router.get("/session_data", response_model=SessionDataResponse)
async def get_session_data(session_data: Tuple[str, dict] = Depends(get_session)):
    """获取当前会话数据"""
    session_id, session = session_data
    return {
        "session_id": session_id,
        "qa_pairs": session["qa_pairs"],
        "history": session["history"],
        "total_count": len(session["qa_pairs"])
    }

@router.delete("/clear_session")
async def clear_session(session_data: Tuple[str, dict] = Depends(get_session)):
    """清空当前会话数据"""
    session_id, session = session_data
    session["qa_pairs"] = []
    session["history"] = []
    return {"message": "Session cleared successfully", "session_id": session_id}
```

### 4.3. 数据模型更新

扩展 QA 对数据模型以支持增量模式的追踪：

```python
class QAPair(BaseModel):
    id: str                # UUID
    question: str          # 问题
    answer: str            # 答案
    chunk: str             # 源文本块
    scores: Score          # 评分
    source: str = "unknown"  # 来源（文件名或"manual_input"）
    batch_id: str = None   # 批次ID
    created_at: datetime = Field(default_factory=datetime.now)

class SessionDataResponse(BaseModel):
    session_id: str
    qa_pairs: List[QAPair]
    history: List[dict]
    total_count: int
```

## 5. 前端实现

### 5.1. 会话状态管理

在前端添加状态管理逻辑以支持增量模式：

```jsx
// 会话管理相关状态
const [sessionId, setSessionId] = useState(null);
const [allQAPairs, setAllQAPairs] = useState([]);
const [history, setHistory] = useState([]);
const [filteredQAPairs, setFilteredQAPairs] = useState([]);

// 会话过滤器状态
const [batchFilter, setBatchFilter] = useState(null);
const [sourceFilter, setSourceFilter] = useState(null);

// 获取会话数据
const fetchSessionData = async () => {
  try {
    const headers = {};
    if (sessionId) {
      headers["X-Session-ID"] = sessionId;
    }
    
    const response = await axios.get('/api/session_data', { headers });
    
    if (response.data) {
      setSessionId(response.data.session_id);
      setAllQAPairs(response.data.qa_pairs);
      setHistory(response.data.history);
      applyFilters(response.data.qa_pairs);
    }
  } catch (err) {
    setError('获取会话数据失败：' + (err.response?.data?.detail || err.message));
  }
};

// 应用过滤器
const applyFilters = (qaPairs) => {
  let filtered = [...qaPairs];
  
  if (batchFilter) {
    filtered = filtered.filter(qa => qa.batch_id === batchFilter);
  }
  
  if (sourceFilter) {
    filtered = filtered.filter(qa => qa.source === sourceFilter);
  }
  
  setFilteredQAPairs(filtered);
};

// 组件初始化时加载会话
useEffect(() => {
  fetchSessionData();
}, []);

// 当过滤器变化时重新应用
useEffect(() => {
  applyFilters(allQAPairs);
}, [batchFilter, sourceFilter, allQAPairs]);
```

### 5.2. 增量处理文件上传

```jsx
const handleIncrementalUpload = async () => {
  if (!fileList.length) return;
  
  setIsUploading(true);
  
  try {
    const formData = new FormData();
    fileList.forEach(file => {
      formData.append('files', file.originFileObj);
    });
    
    const headers = {};
    if (sessionId) {
      headers["X-Session-ID"] = sessionId;
    }
    
    const response = await axios.post('/api/upload_incremental', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        ...headers
      }
    });
    
    if (response.data) {
      setSessionId(response.data.session_id);
      message.success(`成功处理 ${response.data.qa_pairs.length} 个QA对，总计 ${response.data.total_count} 个`);
      fetchSessionData();
    }
  } catch (err) {
    setError('增量处理文件失败：' + (err.response?.data?.detail || err.message));
  } finally {
    setIsUploading(false);
    setFileList([]);
  }
};
```

### 5.3. 增量处理文本输入

```jsx
const handleIncrementalTextProcess = async () => {
  if (!text.trim()) return;
  
  setIsProcessing(true);
  
  try {
    const headers = {};
    if (sessionId) {
      headers["X-Session-ID"] = sessionId;
    }
    
    const response = await axios.post('/api/text_incremental', 
      { text },
      { headers }
    );
    
    if (response.data) {
      setSessionId(response.data.session_id);
      message.success(`成功处理 ${response.data.qa_pairs.length} 个QA对，总计 ${response.data.total_count} 个`);
      fetchSessionData();
    }
  } catch (err) {
    setError('增量处理文本失败：' + (err.response?.data?.detail || err.message));
  } finally {
    setIsProcessing(false);
    setText('');
  }
};
```

### 5.4. 批次和来源筛选器

```jsx
// 生成批次选项
const batchOptions = useMemo(() => {
  const batches = new Set(allQAPairs.map(qa => qa.batch_id));
  return Array.from(batches).map((batchId, index) => ({
    label: `批次 ${index + 1}`,
    value: batchId
  }));
}, [allQAPairs]);

// 生成来源选项
const sourceOptions = useMemo(() => {
  const sources = new Set(allQAPairs.map(qa => qa.source));
  return Array.from(sources).map(source => ({
    label: source === 'manual_input' ? '手动输入' : source,
    value: source
  }));
}, [allQAPairs]);

// 筛选器组件
<div className="filters">
  <Space>
    <Button 
      type={!batchFilter && !sourceFilter ? 'primary' : 'default'} 
      onClick={() => {
        setBatchFilter(null);
        setSourceFilter(null);
      }}
    >
      查看全部
    </Button>
    
    <Select
      placeholder="按批次筛选"
      value={batchFilter}
      onChange={setBatchFilter}
      allowClear
      style={{ width: 150 }}
      options={batchOptions}
    />
    
    <Select
      placeholder="按来源筛选"
      value={sourceFilter}
      onChange={setSourceFilter}
      allowClear
      style={{ width: 150 }}
      options={sourceOptions}
    />
  </Space>
</div>
```

## 6. 处理历史记录组件

```jsx
const HistoryTable = ({ history }) => (
  <Table
    dataSource={history}
    rowKey={(record, index) => index}
    columns={[
      {
        title: '来源',
        dataIndex: 'source',
        key: 'source',
        render: (source) => source === 'manual_input' ? '手动输入' : source,
      },
      {
        title: '类型',
        dataIndex: 'type',
        key: 'type',
        render: (type) => type === 'file' ? '文件' : '文本',
      },
      {
        title: '时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (timestamp) => new Date(timestamp).toLocaleString(),
      },
      {
        title: 'QA对数量',
        dataIndex: 'count',
        key: 'count',
      },
      {
        title: '操作',
        key: 'action',
        render: (_, record, index) => (
          <Button 
            type="link" 
            onClick={() => {
              const targetBatchId = allQAPairs.find(
                qa => qa.source === record.source && 
                new Date(qa.created_at).getTime() === new Date(record.timestamp).getTime()
              )?.batch_id;
              
              if (targetBatchId) {
                setBatchFilter(targetBatchId);
              }
            }}
          >
            查看QA对
          </Button>
        ),
      },
    ]}
  />
);
```

## 7. 注意事项和优化

### 7.1. 去重机制

为避免在增量模式下产生重复QA对，应实现问题相似度检测：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DuplicateDetector:
    def __init__(self, threshold=0.85):
        self.vectorizer = TfidfVectorizer()
        self.threshold = threshold
        self.questions = []
        self.vectors = None
    
    def add_questions(self, questions):
        """添加新问题到检测器"""
        if not questions:
            return
            
        if not self.questions:
            self.questions = questions
            self.vectors = self.vectorizer.fit_transform(questions)
            return
            
        # 为新问题计算向量
        new_vectors = self.vectorizer.transform(questions)
        
        # 更新问题列表和向量
        self.questions.extend(questions)
        if self.vectors is not None:
            self.vectors = np.vstack((self.vectors, new_vectors))
        else:
            self.vectors = new_vectors
    
    def detect_duplicates(self, new_questions):
        """检测新问题中的重复项"""
        if not new_questions or not self.questions:
            return [False] * len(new_questions)
            
        # 计算新问题的向量
        new_vectors = self.vectorizer.transform(new_questions)
        
        # 计算相似度
        similarities = cosine_similarity(new_vectors, self.vectors)
        
        # 判断是否为重复
        is_duplicate = []
        for sim_row in similarities:
            is_duplicate.append(np.max(sim_row) >= self.threshold)
            
        return is_duplicate
```

### 7.2. 会话持久化

为了防止浏览器刷新导致会话丢失，应使用本地存储：

```jsx
// 保存会话ID到本地存储
useEffect(() => {
  if (sessionId) {
    localStorage.setItem('qaGenerator_sessionId', sessionId);
  }
}, [sessionId]);

// 组件初始化时尝试恢复会话
useEffect(() => {
  const savedSessionId = localStorage.getItem('qaGenerator_sessionId');
  if (savedSessionId) {
    setSessionId(savedSessionId);
  }
  
  fetchSessionData();
}, []);
```

### 7.3. 性能优化

对于大量QA对的情况，应当采用分页或虚拟列表：

```jsx
<Table
  dataSource={filteredQAPairs}
  rowKey={(record) => record.id}
  pagination={{
    pageSize: 10,
    showSizeChanger: true,
    pageSizeOptions: ['10', '20', '50', '100'],
    showTotal: (total) => `共 ${total} 个QA对`
  }}
  // ... 其他表格配置
/>
```

## 8. 使用场景示例

### 8.1. 多文档QA知识库构建

1. 用户上传第一批markdown文档
2. 系统生成第一批QA对并显示
3. 用户审核和编辑第一批QA对
4. 用户上传第二批markdown文档
5. 系统生成第二批QA对并添加到现有集合
6. 用户根据需要筛选查看特定来源或批次的QA对
7. 用户完成所有审核后导出完整的QA知识库

### 8.2. 文档更新场景

1. 用户已经处理过一批文档并生成了QA对
2. 文档内容有更新，用户需要反映这些变化
3. 用户上传更新后的markdown文档
4. 系统生成新的QA对，通过去重机制避免完全重复的问题
5. 用户可以通过批次筛选查看最新生成的QA对，审核并编辑
6. 导出包含更新的完整QA知识库

## 9. 最佳实践

1. **先小批量测试**：首次使用时，建议先上传少量文档测试系统的QA对生成质量
2. **定期导出**：在长时间操作过程中，建议定期导出当前QA集合，防止会话意外丢失
3. **按批次审核**：生成新批次后立即审核，而不是等所有文档都处理完毕
4. **利用筛选器**：善用批次和来源筛选器，集中处理相关QA对
5. **注意文件大小**：虽然系统支持增量处理，但单个markdown文件不宜过大，建议控制在合理大小