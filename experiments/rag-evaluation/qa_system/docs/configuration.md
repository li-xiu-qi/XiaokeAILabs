# 配置管理

## 1. 环境变量配置

系统使用环境变量来管理关键配置，如 API 密钥和端点。这些环境变量通过 Pydantic 的 BaseSettings 类进行管理和加载。

### 1.1. 关键环境变量

| 环境变量名 | 描述 | 默认值 | 是否必需 |
|------------|------|---------|----------|
| `SILICONFLOW_API_KEY` | 硅基流动 API 密钥 | 无 | 是 |
| `SILICONFLOW_API_ENDPOINT` | 硅基流动 API 端点 | 无特定默认值 | 是 |
| `LLM_MODEL_NAME` | 使用的语言模型名称 | `THUDM/GLM-4-32B-0414` | 否 |
| `PORT` | FastAPI 服务运行端口 | `8000` | 否 |
| `HOST` | FastAPI 服务运行主机 | `0.0.0.0` | 否 |
| `DEBUG` | 调试模式开关 | `False` | 否 |
| `ALLOW_ORIGINS` | CORS 允许的源，用逗号分隔 | `http://localhost:3000` | 否 |

### 1.2. 环境变量设置方式

环境变量可以通过以下方式设置：

1. **操作系统级别设置**：

   ```bash
   # Linux/macOS
   export SILICONFLOW_API_KEY="your_api_key"
   export SILICONFLOW_API_ENDPOINT="https://your-api-endpoint.com"
   
   # Windows (CMD)
   set SILICONFLOW_API_KEY=your_api_key
   set SILICONFLOW_API_ENDPOINT=https://your-api-endpoint.com
   
   # Windows (PowerShell)
   $env:SILICONFLOW_API_KEY = "your_api_key"
   $env:SILICONFLOW_API_ENDPOINT = "https://your-api-endpoint.com"
   ```

2. **使用 .env 文件**：

   在项目根目录创建 `.env` 文件：
   ```
   SILICONFLOW_API_KEY=your_api_key
   SILICONFLOW_API_ENDPOINT=https://your-api-endpoint.com
   PORT=8000
   HOST=0.0.0.0
   DEBUG=True
   ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173
   ```

## 2. 路由与端口配置

### 2.1. 后端服务端口

FastAPI 后端默认运行在 `8000` 端口。可以通过环境变量 `PORT` 进行修改。

### 2.2. API 路由

后端 API 提供以下主要路由：

| 路由 | 方法 | 描述 | 请求体 | 响应 |
|------|------|------|--------|------|
| `/upload` | POST | 上传文件并生成 QA 对 | 表单数据，包含文件 | JSON 对象，包含生成的 QA 对及评分 |
| `/generate_from_text` | POST | 从文本生成 QA 对 | `{"text": "文本内容"}` | JSON 对象，包含生成的 QA 对及评分 |
| `/submit_review` | POST | 提交审核后的 QA 对 | `{"qa_pairs": [...]}` | 成功/失败响应 |
| `/export_json` | GET | 导出 JSON 格式的 QA 对 | 无 | JSON 文件下载 |
| `/health` | GET | 健康检查端点 | 无 | `{"status": "ok"}` |

### 2.3. CORS 配置

系统使用 FastAPI 的 CORSMiddleware 中间件处理跨源资源共享（CORS）。默认允许来自 `http://localhost:3000` 的请求，可通过环境变量 `ALLOW_ORIGINS` 配置。

示例配置：

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import os

app = FastAPI()

# 从环境变量中获取允许的源，使用逗号分隔
allowed_origins = os.getenv("ALLOW_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 3. 配置管理实现

### 3.1. Pydantic BaseSettings 配置类

使用 Pydantic 的 BaseSettings 类管理全局配置：

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # API 配置
    siliconflow_api_key: str
    siliconflow_api_endpoint: str
    llm_model_name: str = "THUDM/GLM-4-32B-0414"
    
    # 服务器配置
    port: int = 8000
    host: str = "0.0.0.0"
    debug: bool = False
    
    # CORS 配置
    allow_origins: str = "http://localhost:3000"
    
    # Pydantic V2 风格，用于从 .env 文件加载配置
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

# 创建全局配置实例
settings = Settings()
```

### 3.2. 在应用中使用配置

在 FastAPI 应用中使用全局配置：

```python
from fastapi import FastAPI
from .settings import settings

app = FastAPI(
    title="QA Generation API",
    description="API for generating Q&A pairs from text",
    version="1.0.0",
    debug=settings.debug
)

@app.get("/")
async def root():
    return {"message": "Welcome to QA Generation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.host, 
        port=settings.port, 
        reload=settings.debug
    )
```

在 agently 工作流中使用全局配置：

```python
import agently
from .settings import settings

def setup_agent():
    agent = agently.Agent()
    agent.set_llm_provider("openai")
    agent.set_settings("base_url", settings.siliconflow_api_endpoint)
    agent.set_settings("api_key", settings.siliconflow_api_key)
    agent.set_llm_model(settings.llm_model_name)
    return agent
```