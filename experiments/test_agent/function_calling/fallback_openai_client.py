# -*- coding: utf-8 -*-
import asyncio
from typing import Optional, Any, Mapping, Dict
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError, APIError
from openai.types.chat import ChatCompletion

class AsyncFallbackOpenAIClient:
    """
    一个支持备用 API 自动切换的异步 OpenAI 客户端。
    当主 API 调用因特定错误（如内容过滤）失败时，会自动尝试使用备用 API。
    """
    def __init__(
        self,
        primary_api_key: str,
        primary_base_url: str,
        primary_model_name: str,
        fallback_api_key: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
        primary_client_args: Optional[Dict[str, Any]] = None,
        fallback_client_args: Optional[Dict[str, Any]] = None,
        content_filter_error_code: str = "1301", # 特定于 Zhipu 的内容过滤错误代码
        content_filter_error_field: str = "contentFilter", # 特定于 Zhipu 的内容过滤错误字段
        max_retries_primary: int = 1, # 主API重试次数
        max_retries_fallback: int = 1, # 备用API重试次数
        retry_delay_seconds: float = 1.0 # 重试延迟时间
    ):
        """
        初始化 AsyncFallbackOpenAIClient。

        Args:
            primary_api_key: 主 API 的密钥。
            primary_base_url: 主 API 的基础 URL。
            primary_model_name: 主 API 使用的模型名称。
            fallback_api_key: 备用 API 的密钥 (可选)。
            fallback_base_url: 备用 API 的基础 URL (可选)。
            fallback_model_name: 备用 API 使用的模型名称 (可选)。
            primary_client_args: 传递给主 AsyncOpenAI 客户端的其他参数。
            fallback_client_args: 传递给备用 AsyncOpenAI 客户端的其他参数。
            content_filter_error_code: 触发回退的内容过滤错误的特定错误代码。
            content_filter_error_field: 触发回退的内容过滤错误中存在的字段名。
            max_retries_primary: 主 API 失败时的最大重试次数。
            max_retries_fallback: 备用 API 失败时的最大重试次数。
            retry_delay_seconds: 重试前的延迟时间（秒）。
        """
        if not primary_api_key or not primary_base_url:
            raise ValueError("主 API 密钥和基础 URL 不能为空。")

        _primary_args = primary_client_args or {}
        self.primary_client = AsyncOpenAI(api_key=primary_api_key, base_url=primary_base_url, **_primary_args)
        self.primary_model_name = primary_model_name

        self.fallback_client: Optional[AsyncOpenAI] = None
        self.fallback_model_name: Optional[str] = None
        if fallback_api_key and fallback_base_url and fallback_model_name:
            _fallback_args = fallback_client_args or {}
            self.fallback_client = AsyncOpenAI(api_key=fallback_api_key, base_url=fallback_base_url, **_fallback_args)
            self.fallback_model_name = fallback_model_name
        else:
            print("⚠️ 警告: 未完全配置备用 API 客户端。如果主 API 失败，将无法进行回退。")

        self.content_filter_error_code = content_filter_error_code
        self.content_filter_error_field = content_filter_error_field
        self.max_retries_primary = max_retries_primary
        self.max_retries_fallback = max_retries_fallback
        self.retry_delay_seconds = retry_delay_seconds
        self._closed = False

    async def _attempt_api_call(
        self,
        client: AsyncOpenAI,
        model_name: str,
        messages: list[Mapping[str, Any]],
        max_retries: int,
        api_name: str,
        **kwargs: Any
    ) -> ChatCompletion:
        """
        尝试调用指定的 OpenAI API 客户端，并进行重试。
        """
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # print(f"尝试使用 {api_name} API ({client.base_url}) 模型: {kwargs.get('model', model_name)}, 第 {attempt + 1} 次尝试")
                completion = await client.chat.completions.create(
                    model=kwargs.pop('model', model_name),
                    messages=messages,
                    **kwargs
                )
                return completion
            except (APIConnectionError, APITimeoutError) as e: # 通常可以重试的网络错误
                last_exception = e
                print(f"⚠️ {api_name} API 调用时发生可重试错误 ({type(e).__name__}): {e}. 尝试次数 {attempt + 1}/{max_retries + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(self.retry_delay_seconds * (attempt + 1)) # 增加延迟
                else:
                    print(f"❌ {api_name} API 在达到最大重试次数后仍然失败。")
            except APIStatusError as e: # API 返回的特定状态码错误
                is_content_filter_error = False
                if e.status_code == 400:
                    try:
                        error_json = e.response.json()
                        error_details = error_json.get("error", {})
                        if (error_details.get("code") == self.content_filter_error_code and
                            self.content_filter_error_field in error_json):
                            is_content_filter_error = True
                    except Exception:
                        pass # 解析错误响应失败，不认为是内容过滤错误
                
                if is_content_filter_error and api_name == "主": # 如果是主 API 的内容过滤错误，则直接抛出以便回退
                    raise e 
                
                last_exception = e
                print(f"⚠️ {api_name} API 调用时发生 APIStatusError ({e.status_code}): {e}. 尝试次数 {attempt + 1}/{max_retries + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(self.retry_delay_seconds * (attempt + 1))
                else:
                    print(f"❌ {api_name} API 在达到最大重试次数后仍然失败 (APIStatusError)。")
            except APIError as e: # 其他不可轻易重试的 OpenAI 错误
                last_exception = e
                print(f"❌ {api_name} API 调用时发生不可重试错误 ({type(e).__name__}): {e}")
                break # 不再重试此类错误
        
        if last_exception:
            raise last_exception
        raise RuntimeError(f"{api_name} API 调用意外失败。") # 理论上不应到达这里

    async def chat_completions_create(
        self,
        messages: list[Mapping[str, Any]],
        **kwargs: Any  # 用于传递其他 OpenAI 参数，如 max_tokens, temperature 等。
    ) -> ChatCompletion:
        """
        使用主 API 创建聊天补全，如果发生特定内容过滤错误或主 API 调用失败，则回退到备用 API。
        支持对主 API 和备用 API 的可重试错误进行重试。

        Args:
            messages: OpenAI API 的消息列表。
            **kwargs: 传递给 OpenAI API 调用的其他参数。

        Returns:
            ChatCompletion 对象。

        Raises:
            APIError: 如果主 API 和备用 API (如果尝试) 都返回 API 错误。
            RuntimeError: 如果客户端已关闭。
        """
        if self._closed:
            raise RuntimeError("客户端已关闭。")
            
        try:
            completion = await self._attempt_api_call(
                client=self.primary_client,
                model_name=self.primary_model_name,
                messages=messages,
                max_retries=self.max_retries_primary,
                api_name="主",
                **kwargs.copy()
            )
            return completion
        except APIStatusError as e_primary:
            is_content_filter_error = False
            if e_primary.status_code == 400:
                try:
                    error_json = e_primary.response.json()
                    error_details = error_json.get("error", {})
                    if (error_details.get("code") == self.content_filter_error_code and
                        self.content_filter_error_field in error_json):
                        is_content_filter_error = True
                except Exception:
                    pass 
            
            if is_content_filter_error and self.fallback_client and self.fallback_model_name:
                print(f"ℹ️ 主 API 内容过滤错误 ({e_primary.status_code})。尝试切换到备用 API ({self.fallback_client.base_url})...")
                try:
                    fallback_completion = await self._attempt_api_call(
                        client=self.fallback_client,
                        model_name=self.fallback_model_name,
                        messages=messages,
                        max_retries=self.max_retries_fallback,
                        api_name="备用",
                        **kwargs.copy()
                    )
                    print(f"✅ 备用 API 调用成功。")
                    return fallback_completion
                except APIError as e_fallback:
                    print(f"❌ 备用 API 调用最终失败: {type(e_fallback).__name__} - {e_fallback}")
                    raise e_fallback 
            else:
                if not (self.fallback_client and self.fallback_model_name and is_content_filter_error):
                     # 如果不是内容过滤错误，或者没有可用的备用API，则记录主API的原始错误
                    print(f"ℹ️ 主 API 错误 ({type(e_primary).__name__}: {e_primary}), 且不满足备用条件或备用API未配置。")
                raise e_primary
        except APIError as e_primary_other: 
            print(f"❌ 主 API 调用最终失败 (非内容过滤，错误类型: {type(e_primary_other).__name__}): {e_primary_other}")
            if self.fallback_client and self.fallback_model_name:
                print(f"ℹ️ 主 API 失败，尝试切换到备用 API ({self.fallback_client.base_url})...")
                try:
                    fallback_completion = await self._attempt_api_call(
                        client=self.fallback_client,
                        model_name=self.fallback_model_name,
                        messages=messages,
                        max_retries=self.max_retries_fallback,
                        api_name="备用",
                        **kwargs.copy()
                    )
                    print(f"✅ 备用 API 调用成功。")
                    return fallback_completion
                except APIError as e_fallback_after_primary_fail:
                    print(f"❌ 备用 API 在主 API 失败后也调用失败: {type(e_fallback_after_primary_fail).__name__} - {e_fallback_after_primary_fail}")
                    raise e_fallback_after_primary_fail 
            else: 
                raise e_primary_other

    async def close(self):
        """异步关闭主客户端和备用客户端 (如果存在)。"""
        if not self._closed:
            await self.primary_client.close()
            if self.fallback_client:
                await self.fallback_client.close()
            self._closed = True
            # print("AsyncFallbackOpenAIClient 已关闭。")

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("AsyncFallbackOpenAIClient 不能在关闭后重新进入。请创建一个新实例。")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
