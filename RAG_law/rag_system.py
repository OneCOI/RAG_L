"""RAG 系统初始化模块（使用 RAG 组件）"""

import streamlit as st

from components.llm import get_llm
from components.tools import get_tools
from components.agent import create_agent_executor, create_agent_with_history
from knowledge_base import load_legal_knowledge_base


@st.cache_resource
def initialize_rag_system(use_rag: bool, kb_name: str, rebuild_index: bool = False):
    """
    初始化 RAG 系统

    整合了以下组件：
    1. LLM (llm)
    2. 工具 (tools)
    3. Agent (agent)

    Args:
        use_rag: 是否启用 RAG（法律知识库检索）
        kb_name: 知识库名称
        rebuild_index: 是否重建索引

    Returns:
        (agent_with_history, store, tools)
    """
    try:
        # 1. 获取工具
        retriever = None
        if use_rag:
            retriever, db = load_legal_knowledge_base(kb_name, rebuild_index=rebuild_index)
            if retriever is None:
                st.warning("⚠️ 法律知识库加载失败，将仅使用网络搜索")
                use_rag = False

        tools = get_tools(use_rag=use_rag, retriever=retriever)

        # 2. 获取大语言模型
        llm = get_llm()

        # 3. 创建 Agent Executor
        agent_executor = create_agent_executor(llm, tools)

        # 4. 创建带历史记录的 Agent
        agent_with_history, store = create_agent_with_history(agent_executor)

        return agent_with_history, store, tools

    except Exception as e:
        st.error(f"❌ 初始化失败: {e}")
        return None, None, []
