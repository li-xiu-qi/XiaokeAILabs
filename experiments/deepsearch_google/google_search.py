import requests
import json
import os
import tempfile
import pickle
import pytz
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta


class GoogleSearchClient:
    """Google自定义搜索API客户端"""

    def __init__(
        self, api_key: str = "", search_engine_id: str = "", daily_limit: int = 100
    ):
        """
        初始化Google搜索客户端

        Args:
            api_key: Google API密钥
            search_engine_id: 自定义搜索引擎ID
            daily_limit: 每日请求次数上限，默认为100
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.request_count = 0
        self.daily_request_count = 0
        self.last_request_date = self.get_pacific_time().date()
        self.daily_limit = daily_limit

    def get_pacific_time(self) -> datetime:
        """获取当前太平洋时间"""
        pacific_tz = pytz.timezone("US/Pacific")
        return datetime.now(pytz.utc).astimezone(pacific_tz)

    def search(
        self, query: str, language: str = "zh-CN", num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        执行Google搜索

        Args:
            query: 搜索关键词
            language: 结果语言，默认为简体中文
            num_results: 返回结果数量，默认为10

        Returns:
            搜索结果列表，每个结果包含标题、链接和摘要

        Raises:
            Exception: 当搜索请求失败时抛出
        """
        # 检查是否需要重置每日计数（使用太平洋时间）
        pacific_today = self.get_pacific_time().date()
        if pacific_today != self.last_request_date:
            print(
                f"检测到日期变更（太平洋时间）: {self.last_request_date} -> {pacific_today}，重置每日计数"
            )
            self.daily_request_count = 0
            self.last_request_date = pacific_today

        # 检查是否已达到每日限制
        if self.daily_request_count >= self.daily_limit:
            raise Exception(f"已达到每日请求上限 ({self.daily_limit}次)")

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "hl": language,
            "num": min(num_results, 10),  # Google API每次最多返回10条结果
        }

        print(f"正在搜索: '{query}'...")

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            search_results = response.json()

            # 更新请求计数
            self.request_count += 1
            self.daily_request_count += 1

            formatted_results = []
            if "items" in search_results:
                for item in search_results["items"]:
                    formatted_results.append(
                        {
                            "title": item.get("title", "N/A"),
                            "link": item.get("link", "N/A"),
                            "snippet": item.get("snippet", "N/A").replace(chr(10), " "),
                        }
                    )
                return formatted_results
            else:
                print("未找到搜索结果")
                print("原始回复:", search_results)
                return []

        except requests.exceptions.RequestException as e:
            print(f"搜索请求出错: {e}")
            try:
                error_details = response.json()
                print("错误详情:", error_details)
            except (json.JSONDecodeError, UnboundLocalError):
                if "response" in locals():
                    print("服务器原始响应:", response.text)
            raise Exception(f"Google搜索失败: {str(e)}")

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """打印搜索结果"""
        if not results:
            print("没有找到搜索结果。")
            return

        print(f"\n找到 {len(results)} 条搜索结果:\n")
        for i, result in enumerate(results):
            print(f"--- 结果 {i+1} ---")
            print(f"标题: {result['title']}")
            print(f"链接: {result['link']}")
            print(f"摘要: {result['snippet']}")
            print("-" * 20 + "\n")

    def get_client_id(self) -> str:
        """获取客户端的唯一标识符"""
        return f"{self.api_key}_{self.search_engine_id}"

    def is_limit_reached(self) -> bool:
        """检查是否达到每日请求上限（基于太平洋时间）"""
        # 检查是否需要重置每日计数
        pacific_today = self.get_pacific_time().date()
        if pacific_today != self.last_request_date:
            self.daily_request_count = 0
            self.last_request_date = pacific_today
            return False

        return self.daily_request_count >= self.daily_limit

    def get_remaining_requests(self) -> int:
        """获取今日剩余请求次数（基于太平洋时间）"""
        # 检查是否需要重置每日计数
        pacific_today = self.get_pacific_time().date()
        if pacific_today != self.last_request_date:
            return self.daily_limit

        return max(0, self.daily_limit - self.daily_request_count)


class GoogleSearchPool:
    """管理多个Google搜索客户端的池，支持在API额度限制时自动切换"""

    USAGE_FILE = os.path.join(tempfile.gettempdir(), "google_search_usage.pkl")

    def __init__(self):
        """初始化搜索客户端池"""
        self.clients = []  # 存储(api_key, search_engine_id, client)元组
        self.current_index = 0
        self.usage_data = {}  # 存储使用情况数据

        # 尝试加载使用情况数据
        self._load_usage_data()

    def add_client(
        self, api_key: str, search_engine_id: str, daily_limit: int = 100
    ) -> None:
        """
        添加一个新的API密钥和搜索引擎ID组合到池中

        Args:
            api_key: Google API密钥
            search_engine_id: 自定义搜索引擎ID
            daily_limit: 每日请求次数上限，默认为100
        """
        client = GoogleSearchClient(api_key, search_engine_id, daily_limit)
        client_id = client.get_client_id()

        # 如果有历史使用记录，恢复计数
        if client_id in self.usage_data:
            client.request_count = self.usage_data[client_id].get("total_requests", 0)

            # 检查上次请求日期（与太平洋时间比较）
            last_date_str = self.usage_data[client_id].get("last_date")
            if last_date_str:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                pacific_today = client.get_pacific_time().date()
                if last_date == pacific_today:
                    client.daily_request_count = self.usage_data[client_id].get(
                        "daily_requests", 0
                    )
                    client.last_request_date = last_date

            # 恢复每日限制设置
            if "daily_limit" in self.usage_data[client_id]:
                client.daily_limit = self.usage_data[client_id]["daily_limit"]

        self.clients.append((api_key, search_engine_id, client))
        print(f"已添加新的搜索客户端 (总数: {len(self.clients)})")

        if client_id in self.usage_data:
            remaining = client.get_remaining_requests()
            print(
                f"客户端已使用 {client.request_count} 次 (今日: {client.daily_request_count}/{client.daily_limit}，剩余: {remaining}次)"
            )

    def search(
        self, query: str, language: str = "zh-CN", num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        使用池中的客户端执行搜索，遇到错误或达到限制时自动切换到下一个客户端

        Args:
            query: 搜索关键词
            language: 结果语言
            num_results: 希望返回的结果数量

        Returns:
            搜索结果列表

        Raises:
            Exception: 当所有客户端都失败时抛出
        """
        if not self.clients:
            raise Exception("搜索池为空，请先添加搜索客户端")

        # 选择合适的客户端
        self._select_available_client()

        # 尝试所有客户端，直到成功或全部失败
        attempts = 0
        max_attempts = len(self.clients)

        while attempts < max_attempts:
            if not self.clients:
                raise Exception("所有搜索客户端均已失效")

            _, _, client = self.clients[self.current_index]

            # 如果当前客户端已达上限，尝试下一个
            if client.is_limit_reached():
                print(
                    f"客户端 {self.current_index + 1}/{len(self.clients)} 已达每日上限 ({client.daily_limit}次)"
                )
                self._rotate_to_next_client()
                attempts += 1
                continue

            try:
                results = client.search(query, language, num_results)
                # 更新使用情况数据
                self._update_usage_data(client)
                return results
            except Exception as e:
                print(
                    f"客户端 {self.current_index + 1}/{len(self.clients)} 失败: {str(e)}"
                )

                # 处理API配额限制或上限达到的情况
                if "quota" in str(e).lower() or "上限" in str(e):
                    print(f"检测到API配额限制，尝试下一个客户端")
                    self._rotate_to_next_client()
                else:
                    # 其他错误，尝试下一个客户端
                    self._rotate_to_next_client()

                attempts += 1

        raise Exception(f"所有 {max_attempts} 个搜索客户端均尝试失败或已达上限")

    def _rotate_to_next_client(self) -> None:
        """轮换到下一个可用的客户端"""
        if not self.clients:
            return

        self.current_index = (self.current_index + 1) % len(self.clients)

    def _select_client_with_least_usage(self) -> None:
        """选择使用次数最少的客户端"""
        if not self.clients:
            return

        min_requests = float("inf")
        min_index = 0

        for i, (_, _, client) in enumerate(self.clients):
            # 优先考虑每日请求次数，更均衡地使用每个客户端的每日配额
            if client.daily_request_count < min_requests:
                min_requests = client.daily_request_count
                min_index = i

        self.current_index = min_index

    def _select_available_client(self) -> None:
        """选择未达到每日上限且请求次数最少的客户端"""
        if not self.clients:
            return

        # 首先检查是否有任何可用的客户端（未达到上限）
        available_clients = [
            i
            for i, (_, _, client) in enumerate(self.clients)
            if not client.is_limit_reached()
        ]

        if not available_clients:
            print("警告：所有客户端均已达到每日请求上限")
            # 使用原有的选择逻辑，可能会导致异常
            self._select_client_with_least_usage()
            return

        # 在可用客户端中选择每日请求次数最少的
        min_requests = float("inf")
        min_index = available_clients[0]

        for i in available_clients:
            _, _, client = self.clients[i]
            if client.daily_request_count < min_requests:
                min_requests = client.daily_request_count
                min_index = i

        self.current_index = min_index
        _, _, selected_client = self.clients[self.current_index]
        remaining = selected_client.get_remaining_requests()
        print(
            f"已选择客户端 {self.current_index + 1}/{len(self.clients)}，"
            f"今日已使用 {selected_client.daily_request_count}/{selected_client.daily_limit} 次，"
            f"剩余 {remaining} 次请求"
        )

    def _update_usage_data(self, client: GoogleSearchClient) -> None:
        """更新客户端使用情况数据并保存到文件"""
        client_id = client.get_client_id()

        self.usage_data[client_id] = {
            "total_requests": client.request_count,
            "daily_requests": client.daily_request_count,
            "last_date": client.last_request_date.strftime("%Y-%m-%d"),
            "daily_limit": client.daily_limit,
            "pacific_time": client.get_pacific_time().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

        self._save_usage_data()

    def _save_usage_data(self) -> None:
        """保存使用情况数据到临时文件"""
        try:
            with open(self.USAGE_FILE, "wb") as f:
                pickle.dump(self.usage_data, f)
        except Exception as e:
            print(f"保存使用情况数据失败: {str(e)}")

    def _load_usage_data(self) -> None:
        """从临时文件加载使用情况数据"""
        try:
            if os.path.exists(self.USAGE_FILE):
                with open(self.USAGE_FILE, "rb") as f:
                    self.usage_data = pickle.load(f)
                print(
                    f"已加载搜索API使用情况数据，共 {len(self.usage_data)} 个客户端记录"
                )
            else:
                self.usage_data = {}
        except Exception as e:
            print(f"加载使用情况数据失败: {str(e)}")
            self.usage_data = {}

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """打印搜索结果，复用GoogleSearchClient的方法"""
        if self.clients:
            self.clients[0][2].display_results(results)
        else:
            print("没有可用的搜索客户端")

    def display_usage_stats(self) -> None:
        """显示所有客户端的使用统计信息"""
        print("\n客户端使用情况统计:")
        print("-" * 50)
        pacific_time = None

        for i, (api_key, search_engine_id, client) in enumerate(self.clients):
            if pacific_time is None:
                pacific_time = client.get_pacific_time()
                print(
                    f"当前太平洋时间: {pacific_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )
                print("-" * 50)

            masked_key = f"{api_key[:5]}...{api_key[-3:]}"
            print(f"客户端 {i+1}: {masked_key} | {search_engine_id}")
            print(f"  总请求次数: {client.request_count}")
            remaining = client.get_remaining_requests()
            print(
                f"  今日请求次数: {client.daily_request_count}/{client.daily_limit} (剩余: {remaining})"
            )
            print(f"  最后请求日期: {client.last_request_date} (太平洋时间)")
            print("-" * 30)


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # 创建搜索池实例
    search_pool = GoogleSearchPool()

    # 添加多个API密钥和搜索引擎ID组合，并设置不同的每日限制
    # 从环境变量读取第一个客户端的凭据
    api_key_1 = os.getenv("GOOGLE_API_KEY")
    engine_id_1 = os.getenv("GOOGLE_ENGINE_ID")

    if api_key_1 and engine_id_1:
        search_pool.add_client(api_key_1, engine_id_1, daily_limit=100)
    else:
        print(
            "警告：未在环境变量中找到 GOOGLE_API_KEY 和 GOOGLE_ENGINE_ID，无法添加默认客户端。"
        )

    # 示例：添加第二个客户端（如果环境变量中定义了）
    # api_key_2 = os.getenv("GOOGLE_API_KEY_2")
    # engine_id_2 = os.getenv("GOOGLE_ENGINE_ID_2")
    # if api_key_2 and engine_id_2:
    #     search_pool.add_client(api_key_2, engine_id_2, daily_limit=50)
    # else:
    #     print("提示：未找到 GOOGLE_API_KEY_2 和 GOOGLE_ENGINE_ID_2，跳过添加第二个客户端。")

    # 也可以统一设置所有客户端的限制
    # search_pool.set_daily_limit_for_all(100)

    # 执行搜索，会自动使用未达到限制的客户端
    if not search_pool.clients:
        print("错误：搜索池中没有可用的客户端。请检查您的环境变量配置。")
    else:
        query = "卢森堡有什么好玩的"
        try:
            results = search_pool.search(query)
            search_pool.display_results(results)

            # 显示使用情况统计
            search_pool.display_usage_stats()
        except Exception as e:
            print(f"搜索失败: {str(e)}")
