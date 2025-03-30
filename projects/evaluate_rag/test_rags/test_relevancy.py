from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy
from datasets import Dataset
from ragas.metrics import context_recall
from ragas import evaluate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import asyncio

# 加载环境变量
load_dotenv()

# 从环境变量获取API密钥和基础URL
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")

# 创建LLM实例
llm = ChatOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    model="Pro/deepseek-ai/DeepSeek-V3"
)

# 创建Embedding模型实例
embeddings = OpenAIEmbeddings(
    api_key=openai_api_key,
    base_url=openai_api_base,
    model="BAAI/bge-m3"
)

def main():
    # 创建单个样本
    sample = SingleTurnSample(
            user_input="第一届超级碗是什么时候？",
            response="第一届超级碗于1967年1月15日举行",
            retrieved_contexts=[
                "第一届AFL-NFL世界冠军赛是1967年1月15日在洛杉矶洛杉矶纪念体育馆举行的一场美式足球比赛。"
            ]
        )
    
    # 将样本转换为Dataset对象
    dataset = Dataset.from_dict({
        "question": [sample.user_input],
        "answer": [sample.response],
        "contexts": [sample.retrieved_contexts]
    })
    
    # 使用evaluate函数评估
    metrics = [ResponseRelevancy(llm=llm, embeddings=embeddings)]
    result = evaluate(dataset, metrics)
    print(result)

# 对于同步运行，我们不需要asyncio.run
if __name__ == "__main__":
    # ragas的evaluate函数是同步的，所以我们移除async/await
    main()