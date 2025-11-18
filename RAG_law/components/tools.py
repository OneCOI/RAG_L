"""工具组件"""

from typing import List, Optional

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_core.retrievers import BaseRetriever

from config import TAVILY_MAX_RESULTS, LEGAL_TOOL_NAME, LEGAL_TOOL_DESCRIPTION


def create_search_tool(max_results: int = None) -> TavilySearchResults:
    """
    创建 Tavily 网络搜索工具

    Args:
        max_results: 最大结果数

    Returns:
        TavilySearchResults 工具实例
    """
    if max_results is None:
        max_results = TAVILY_MAX_RESULTS
    return TavilySearchResults(max_results=max_results)


def create_legal_retriever_tool(
    retriever: BaseRetriever,
    name: str = None,
    description: str = None,
) -> BaseTool:
    """
    创建法律知识库检索工具

    Args:
        retriever: 检索器实例
        name: 工具名称
        description: 工具描述

    Returns:
        检索工具实例
    """
    if name is None:
        name = LEGAL_TOOL_NAME
    if description is None:
        description = LEGAL_TOOL_DESCRIPTION

    return create_retriever_tool(
        retriever=retriever,
        name=name,
        description=description,
    )


def get_tools(
    use_rag: bool = True,
    retriever: Optional[BaseRetriever] = None,
) -> List[BaseTool]:
    """
    获取所有工具列表

    Args:
        use_rag: 是否启用 RAG（法律知识库检索）
        retriever: 检索器实例（如果启用 RAG）

    Returns:
        工具列表
    """
    tools = [create_search_tool()]

    if use_rag and retriever is not None:
        legal_tool = create_legal_retriever_tool(retriever)
        tools.append(legal_tool)

    return tools

