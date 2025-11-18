"""大语言模型组件"""

import os
from langchain_openai import ChatOpenAI

from config import (
    DASHSCOPE_API_KEY,
    LLM_MODEL_NAME,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


def get_llm(
    model_name: str = None,
    base_url: str = None,
    api_key: str = None,
    **kwargs
) -> ChatOpenAI:
    """
    获取大语言模型实例

    Args:
        model_name: 模型名称
        base_url: API 基础 URL
        api_key: API Key，默认为配置中的值
        **kwargs: 其他传递给 ChatOpenAI 的参数

    Returns:
        ChatOpenAI 实例
    """
    if model_name is None:
        model_name = LLM_MODEL_NAME
    if base_url is None:
        base_url = LLM_BASE_URL
    if api_key is None:
        api_key = DASHSCOPE_API_KEY

    # 应用配置中的默认参数
    if LLM_TEMPERATURE is not None and "temperature" not in kwargs:
        kwargs["temperature"] = LLM_TEMPERATURE
    if LLM_MAX_TOKENS is not None and "max_tokens" not in kwargs:
        kwargs["max_tokens"] = LLM_MAX_TOKENS

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        **kwargs
    )

