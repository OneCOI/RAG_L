"""Agent 组件"""

from typing import Dict, List, Tuple

from langchain_classic import hub
from langchain_classic.agents import (
    create_tool_calling_agent,
    AgentExecutor,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from config import AGENT_VERBOSE, AGENT_RETURN_INTERMEDIATE_STEPS, AGENT_PROMPT_HUB


def create_agent_executor(
    llm: ChatOpenAI,
    tools: List[BaseTool],
    verbose: bool = None,
    return_intermediate_steps: bool = None,
) -> AgentExecutor:
    """
    创建 Agent Executor

    Args:
        llm: 大语言模型实例
        tools: 工具列表
        verbose: 是否输出详细信息
        return_intermediate_steps: 是否返回中间步骤

    Returns:
        AgentExecutor 实例
    """
    if verbose is None:
        verbose = AGENT_VERBOSE
    if return_intermediate_steps is None:
        return_intermediate_steps = AGENT_RETURN_INTERMEDIATE_STEPS

    prompt = hub.pull(AGENT_PROMPT_HUB)
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
    )

    return agent_executor


def create_agent_with_history(
    agent_executor: AgentExecutor,
    store: Dict = None,
) -> Tuple[RunnableWithMessageHistory, Dict]:
    """
    创建带历史记录的 Agent

    Args:
        agent_executor: AgentExecutor 实例
        store: 消息历史存储字典，如果为 None 则创建新字典

    Returns:
        (带历史记录的 Agent, 消息历史存储字典)
    """
    if store is None:
        store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent_with_history = RunnableWithMessageHistory(
        runnable=agent_executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_history, store

