# 后端设计

## 1. 技术栈

- **Web 框架**：FastAPI
- **Agent 框架**：agently
- **语言模型**：THUDM/GLM-4-32B-0414（通过硅基流动 API）
- **配置管理**：Pydantic BaseSettings
- **环境管理**：环境变量，.env 文件

## 2. 后端架构

FastAPI 后端与 agently 工作流相结合，形成完整的 QA 生成和评分系统。后端主要负责以下功能：

1. 处理前端请求（文件上传、文本生成、导出 JSON 等）
2. 调用 agently 工作流进行 QA 生成和评分
3. 配置管理和环境变量加载
4. 错误处理和异常管理

### 2.1. 项目结构

```
qa_system/
├── main.py               # FastAPI 入口点
├── settings.py           # 配置管理
├── models/
│   ├── __init__.py
│   ├── qa.py             # QA 数据模型
│   └── request.py        # 请求数据模型
├── routers/
│   ├── __init__.py
│   ├── qa_generation.py  # QA 生成路由
│   └── export.py         # 导出路由
├── workflows/
│   ├── __init__.py
│   ├── qa_generator.py   # QA 生成工作流
│   └── qa_scoring.py     # QA 评分工作流
├── utils/
│   ├── __init__.py
│   ├── file_parser.py    # 文件解析工具
│   └── text_processor.py # 文本处理工具
└── prompts/
    ├── __init__.py
    ├── generation.py     # QA 生成提示词
    └── scoring.py        # QA 评分提示词
```

## 3. 数据模型

### 3.1. QA 数据模型

使用 Pydantic 模型定义 QA 数据结构：

```python
from pydantic import BaseModel
from typing import Dict, List, Optional

class Score(BaseModel):
    independence: float  # 独立性评分
    usefulness: float    # 有用性评分
    answerability: float # 可回答性评分

class QAPair(BaseModel):
    id: str  # UUID
    question: str  # 问题
    answer: str    # 答案
    chunk: str     # 源文本块
    scores: Score  # 评分

class QAResponse(BaseModel):
    qa_pairs: List[QAPair]  # QA 对列表
    
class QAExport(BaseModel):
    q: str       # 问题
    a: str       # 答案
    chunk: str   # 源文本块
```

### 3.2. 请求数据模型

```python
from pydantic import BaseModel
from typing import List

class TextGenerationRequest(BaseModel):
    text: str  # 输入文本

class ReviewSubmitRequest(BaseModel):
    qa_pairs: List[QAPair]  # 审核后的 QA 对
```

## 4. API 路由设计

### 4.1. QA 生成路由

```python
from fastapi import APIRouter, UploadFile, File, Depends
from ..models.qa import QAResponse
from ..models.request import TextGenerationRequest
from ..workflows.qa_generator import generate_qa_from_text
from ..utils.file_parser import parse_file_to_text

router = APIRouter()

@router.post("/upload", response_model=QAResponse)
async def upload_file(file: UploadFile = File(...)):
    """从上传的文件生成 QA 对"""
    text = await parse_file_to_text(file)
    qa_pairs = await generate_qa_from_text(text)
    return {"qa_pairs": qa_pairs}

@router.post("/generate_from_text", response_model=QAResponse)
async def generate_from_text(request: TextGenerationRequest):
    """从文本生成 QA 对"""
    qa_pairs = await generate_qa_from_text(request.text)
    return {"qa_pairs": qa_pairs}
```

### 4.2. 审核与导出路由

```python
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import json
from ..models.qa import QAResponse, QAExport
from ..models.request import ReviewSubmitRequest

router = APIRouter()

@router.post("/submit_review", response_model=QAResponse)
async def submit_review(request: ReviewSubmitRequest):
    """提交审核后的 QA 对"""
    # 这里可以添加额外的处理逻辑
    return {"qa_pairs": request.qa_pairs}

@router.get("/export_json")
async def export_json():
    """导出 JSON 格式的 QA 对"""
    # 获取最新审核的 QA 对
    # 此处应该从某种存储中获取（内存、数据库等）
    qa_pairs = get_reviewed_qa_pairs()
    
    # 转换为导出格式
    export_data = [
        QAExport(q=qa.question, a=qa.answer, chunk=qa.chunk).dict()
        for qa in qa_pairs
    ]
    
    return JSONResponse(
        content=export_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=qa_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )
```

## 5. agently 工作流

### 5.1. QA 生成工作流

```python
import agently
import uuid
from ..settings import settings
from ..models.qa import QAPair, Score
from ..prompts.generation import QA_GENERATION_PROMPT

async def generate_qa_from_text(text: str) -> List[QAPair]:
    """从文本生成 QA 对"""
    
    # 设置 agently Agent
    agent = setup_qa_generation_agent()
    
    # 如果文本过长，可以进行分块处理
    chunks = chunk_text(text)
    qa_pairs = []
    
    for chunk in chunks:
        # 调用 Agent 生成 QA 对
        response = await agent.run({"context": chunk})
        
        # 解析响应
        qa_pair = parse_qa_response(response, chunk)
        if qa_pair:
            # 进行评分
            qa_pair = await score_qa_pair(qa_pair)
            qa_pairs.append(qa_pair)
    
    return qa_pairs

def setup_qa_generation_agent():
    """设置 QA 生成 Agent"""
    agent = agently.Agent()
    agent.set_llm_provider("openai")
    agent.set_settings("base_url", settings.siliconflow_api_endpoint)
    agent.set_settings("api_key", settings.siliconflow_api_key)
    agent.set_llm_model(settings.llm_model_name)

    # 设置提示词
    agent.set_system_prompt(QA_GENERATION_PROMPT)
    
    return agent

def parse_qa_response(response: str, chunk: str) -> Optional[QAPair]:
    """解析 Agent 返回的响应，提取问题和答案"""
    try:
        # 假设响应格式如下：
        # Output:::
        # Factoid question: [问题]
        # Answer: [答案]
        
        lines = response.strip().split("\n")
        if len(lines) < 2:
            return None
        
        question_line = lines[0]
        answer_line = lines[1]
        
        question = question_line.replace("Factoid question:", "").strip()
        answer = answer_line.replace("Answer:", "").strip()
        
        return QAPair(
            id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            chunk=chunk,
            scores=Score(independence=0.0, usefulness=0.0, answerability=0.0)
        )
    except Exception as e:
        print(f"Error parsing QA response: {e}")
        return None

def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    """将文本分成较小的块（简单实现）"""
    # 实际实现应该更智能，考虑句子和段落边界
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i:i + max_chunk_size])
    
    return chunks
```

### 5.2. QA 评分工作流

```python
from ..prompts.scoring import INDEPENDENCE_PROMPT, USEFULNESS_PROMPT, ANSWERABILITY_PROMPT

async def score_qa_pair(qa_pair: QAPair) -> QAPair:
    """对 QA 对进行评分"""
    # 设置评分 Agent
    scoring_agent = setup_scoring_agent()
    
    # 独立性评分
    independence_score = await get_independence_score(
        scoring_agent, qa_pair.question
    )
    
    # 有用性评分
    usefulness_score = await get_usefulness_score(
        scoring_agent, qa_pair.question
    )
    
    # 可回答性评分
    answerability_score = await get_answerability_score(
        scoring_agent, qa_pair.question, qa_pair.chunk
    )
    
    # 更新分数
    qa_pair.scores = Score(
        independence=independence_score,
        usefulness=usefulness_score,
        answerability=answerability_score
    )
    
    return qa_pair

def setup_scoring_agent():
    """设置评分 Agent"""
    agent = agently.Agent()
    agent.set_llm_provider("openai")
    agent.set_settings("base_url", settings.siliconflow_api_endpoint)
    agent.set_settings("api_key", settings.siliconflow_api_key)
    agent.set_llm_model(settings.llm_model_name)
    
    return agent

async def get_independence_score(agent, question: str) -> float:
    """获取问题独立性评分"""
    agent.set_system_prompt(INDEPENDENCE_PROMPT)
    response = await agent.run({"question": question})
    return parse_score(response)

async def get_usefulness_score(agent, question: str) -> float:
    """获取问题有用性评分"""
    agent.set_system_prompt(USEFULNESS_PROMPT)
    response = await agent.run({"question": question})
    return parse_score(response)

async def get_answerability_score(agent, question: str, context: str) -> float:
    """获取问题可回答性评分"""
    agent.set_system_prompt(ANSWERABILITY_PROMPT)
    response = await agent.run({"question": question, "context": context})
    return parse_score(response)

def parse_score(response: str) -> float:
    """解析评分响应"""
    try:
        # 假设响应格式为一个 0-10 的数字
        score_str = response.strip()
        score = float(score_str)
        # 标准化为 0-1 范围
        return max(0.0, min(1.0, score / 10))
    except ValueError:
        print(f"Error parsing score: {response}")
        return 0.5  # 默认中等分数
```

## 6. 提示词设计

### 6.1. QA 生成提示词

```python
# prompts/generation.py

QA_GENERATION_PROMPT = """
你的任务是根据给定的上下文编写一个事实性问题及其答案。
你的事实性问题应当能够通过上下文中的具体、简洁的事实信息来回答。
你的事实性问题应当以用户在搜索引擎中提问的风格来表述。
这意味着你的事实性问题不能包含"根据文章"或"上下文"等表述。

请按照以下格式提供你的回答：

Output:::
Factoid question: [这里是你的事实性问题]
Answer: [这里是你对该事实性问题的回答]

以下是上下文：

Context: {context}
Output:::
"""
```

### 6.2. QA 评分提示词

```python
# prompts/scoring.py

INDEPENDENCE_PROMPT = """
请评估以下问题的上下文独立性，即问题本身是否不依赖于特定上下文或文档就能理解。
评分标准：0-10分，其中0分表示完全依赖上下文（例如"文中提到的第三个因素是什么？"），
10分表示完全独立（例如"谁是美国第一任总统？"）。

问题: {question}

请只返回一个0-10之间的数字作为评分。
"""

USEFULNESS_PROMPT = """
请评估以下问题对开发者的有用性。
一个有用的问题应当：
1. 针对开发者可能关注的重要概念、功能或问题
2. 能帮助解决实际编程或设计问题
3. 关注关键信息而非琐碎细节

评分标准：0-10分，0分表示完全无用，10分表示非常有用。

问题: {question}

请只返回一个0-10之间的数字作为评分。
"""

ANSWERABILITY_PROMPT = """
根据提供的上下文，评估以下问题的可回答性。
一个高可回答性的问题可以使用上下文中的信息直接且准确地回答。

评分标准：0-10分，0分表示完全无法从上下文回答，10分表示可以从上下文中完全准确地回答。

问题: {question}

上下文:
{context}

请只返回一个0-10之间的数字作为评分。
"""
```

## 7. 错误处理

### 7.1. 异常管理

```python
from fastapi import HTTPException, status

class QAGenerationError(Exception):
    """QA 生成过程中的错误"""
    pass

class FileParsingError(Exception):
    """文件解析错误"""
    pass

class ScoringError(Exception):
    """评分错误"""
    pass

async def parse_file_to_text(file: UploadFile) -> str:
    """解析上传的文件为文本"""
    try:
        content = await file.read()
        if file.filename.endswith('.txt') or file.filename.endswith('.md'):
            return content.decode('utf-8')
        elif file.filename.endswith('.pdf'):
            # 使用适当的PDF解析库
            return parse_pdf(content)
        elif file.filename.endswith('.docx'):
            # 使用适当的DOCX解析库
            return parse_docx(content)
        else:
            raise FileParsingError(f"不支持的文件类型: {file.filename}")
    except Exception as e:
        raise FileParsingError(f"文件解析错误: {str(e)}")

# 在路由中使用异常处理
@router.post("/upload", response_model=QAResponse)
async def upload_file(file: UploadFile = File(...)):
    """从上传的文件生成 QA 对"""
    try:
        text = await parse_file_to_text(file)
        qa_pairs = await generate_qa_from_text(text)
        return {"qa_pairs": qa_pairs}
    except FileParsingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except QAGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA 生成错误: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"意外错误: {str(e)}"
        )
```

### 7.2. 错误日志

```python
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qa_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("qa_system")

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志和处理时间"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Path: {request.url.path} | "
            f"Method: {request.method} | "
            f"Status: {response.status_code} | "
            f"Process Time: {process_time:.3f}s"
        )
        return response
    except Exception as e:
        logger.error(
            f"Error processing request: {str(e)}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"}
        )

# 异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )
```

## 8. 优化与扩展

### 8.1. 并行处理

使用异步和并行处理，提高大文本的处理效率：

```python
import asyncio

async def generate_qa_from_text(text: str) -> List[QAPair]:
    """从文本生成 QA 对（并行处理）"""
    chunks = chunk_text(text)
    
    # 并行处理每个文本块
    tasks = [generate_qa_for_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    # 展平结果列表
    qa_pairs = [pair for sublist in results for pair in sublist]
    return qa_pairs

async def generate_qa_for_chunk(chunk: str) -> List[QAPair]:
    """为单个文本块生成 QA 对"""
    agent = setup_qa_generation_agent()
    response = await agent.run({"context": chunk})
    
    qa_pair = parse_qa_response(response, chunk)
    if qa_pair:
        qa_pair = await score_qa_pair(qa_pair)
        return [qa_pair]
    return []
```

### 8.2. 缓存机制

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def get_cached_qa_pairs(text_hash: str):
    """获取缓存的 QA 对"""
    # 实际实现中，可以从数据库或其他持久化存储获取
    pass

async def generate_qa_from_text(text: str) -> List[QAPair]:
    """从文本生成 QA 对（带缓存）"""
    # 计算文本哈希值
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # 尝试从缓存获取
    cached_result = get_cached_qa_pairs(text_hash)
    if cached_result:
        return cached_result
    
    # 如果缓存未命中，生成新的 QA 对
    # ... 生成逻辑 ...
    
    # 更新缓存
    update_qa_cache(text_hash, qa_pairs)
    
    return qa_pairs
```

### 8.3. 批量评分

```python
async def batch_score_qa_pairs(qa_pairs: List[QAPair]) -> List[QAPair]:
    """批量对 QA 对进行评分"""
    scoring_agent = setup_scoring_agent()
    
    # 创建评分任务
    independence_tasks = [
        get_independence_score(scoring_agent, qa.question)
        for qa in qa_pairs
    ]
    
    usefulness_tasks = [
        get_usefulness_score(scoring_agent, qa.question)
        for qa in qa_pairs
    ]
    
    answerability_tasks = [
        get_answerability_score(scoring_agent, qa.question, qa.chunk)
        for qa in qa_pairs
    ]
    
    # 并行执行评分
    independence_scores = await asyncio.gather(*independence_tasks)
    usefulness_scores = await asyncio.gather(*usefulness_tasks)
    answerability_scores = await asyncio.gather(*answerability_tasks)
    
    # 更新 QA 对的评分
    for i, qa in enumerate(qa_pairs):
        qa.scores = Score(
            independence=independence_scores[i],
            usefulness=usefulness_scores[i],
            answerability=answerability_scores[i]
        )
    
    return qa_pairs
```