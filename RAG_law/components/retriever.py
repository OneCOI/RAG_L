"""检索器组件"""

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from config import RETRIEVER_SEARCH_KWARGS, RETRIEVER_SEARCH_TYPE


def create_retriever(vector_store: FAISS, **kwargs) -> BaseRetriever:
    """
    从向量存储创建检索器

    Args:
        vector_store: FAISS 向量存储实例
        **kwargs: 传递给 as_retriever() 的参数

    Returns:
        检索器实例
    """
    # 应用配置中的默认参数
    if RETRIEVER_SEARCH_KWARGS is not None and "search_kwargs" not in kwargs:
        kwargs["search_kwargs"] = RETRIEVER_SEARCH_KWARGS
    if RETRIEVER_SEARCH_TYPE is not None and "search_type" not in kwargs:
        kwargs["search_type"] = RETRIEVER_SEARCH_TYPE

    return vector_store.as_retriever(**kwargs)

