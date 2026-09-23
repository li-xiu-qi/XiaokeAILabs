import sqlite3
from contextlib import contextmanager
from fastapi import FastAPI, Depends, HTTPException
from typing import Iterator, List, Dict, Any

# 使用 Pydantic V2 的新方式处理设置
try:
    # 尝试导入 pydantic-settings (Pydantic V2)
    from pydantic_settings import BaseSettings
except ImportError:
    # 回退到 Pydantic V1 的导入方式
    from pydantic import BaseSettings

# 配置设置类
class Settings(BaseSettings):
    DB_PATH: str = "test.db"  # 默认数据库路径

    # 更新为与 Pydantic V2 兼容的方法
    @classmethod
    def get_config(cls):
        return cls()

# 创建设置实例
settings = Settings()

# 创建FastAPI应用
app = FastAPI(title="数据库连接依赖注入示例")

@contextmanager  # 将生成器函数转换为上下文管理器，实现自动资源管理
def get_db_connection():
    """获取数据库连接，使用上下文管理器确保连接正确关闭
    
    @contextmanager 装饰器的作用：
    1. 将一个生成器函数转换为上下文管理器，使其可以在 with 语句中使用
    2. 自动处理资源的分配和释放，确保即使发生异常也能正确关闭资源
    3. yield 语句前的代码在进入 with 块时执行（资源获取）
    4. yield 语句后的代码在离开 with 块时执行（资源释放）
    5. 简化了创建上下文管理器的过程，不需要手动实现 __enter__ 和 __exit__ 方法
    
    使用示例：
    with get_db_connection() as conn:
        # 在这里使用数据库连接
        cursor = conn.cursor()
        # ...
    # 离开 with 块后，连接自动关闭
    """
    conn = None
    try:
        conn = sqlite3.connect(settings.get_config().DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()

# FastAPI依赖函数
def get_db() -> Iterator[sqlite3.Connection]:
    """将上下文管理器转换为FastAPI可用的依赖项"""
    with get_db_connection() as conn:
        yield conn

# 示例路由，使用依赖注入获取数据库连接
@app.get("/users/", response_model=List[Dict[str, Any]])
def get_users(db: sqlite3.Connection = Depends(get_db)):
    """获取所有用户的示例路由"""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users")
        users = [dict(row) for row in cursor.fetchall()]
        return users
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"数据库错误: {str(e)}")

# 初始化数据库的路由
@app.on_event("startup")
async def init_db():
    """应用启动时初始化数据库"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 创建用户表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL
            )
            """)
            
            # 插入一些测试数据
            cursor.execute("SELECT count(*) FROM users")
            if cursor.fetchone()[0] == 0:
                users = [
                    ("user1", "user1@example.com"),
                    ("user2", "user2@example.com"),
                    ("user3", "user3@example.com"),
                ]
                cursor.executemany("INSERT INTO users (username, email) VALUES (?, ?)", users)
            
            conn.commit()
            print("数据库初始化完成")
    except sqlite3.Error as e:
        print(f"数据库初始化错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
