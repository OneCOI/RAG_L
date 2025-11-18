from typing import Any, Dict, List
from pathlib import Path
import json
import re

import streamlit as st

from config import CHAT_LOG_DIR


def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 AgentExecutor 的返回结果中提取工具调用信息"""
    tool_calls: List[Dict[str, Any]] = []

    if "intermediate_steps" not in response:
        return tool_calls

    for step in response["intermediate_steps"]:
        if isinstance(step, (list, tuple)) and len(step) >= 1:
            action = step[0]
            if hasattr(action, "tool"):
                tool_name = action.tool
            elif hasattr(action, "name"):
                tool_name = action.name
            elif isinstance(action, dict):
                tool_name = action.get("tool") or action.get("name", "未知工具")
            else:
                tool_name = str(action)

            if tool_name and tool_name != "未知工具":
                existing = next(
                    (tc for tc in tool_calls if tc.get("name") == tool_name), None
                )
                if existing:
                    existing["count"] = existing.get("count", 1) + 1
                else:
                    tool_calls.append({"name": tool_name, "count": 1})

    return tool_calls


def render_tool_calls(tool_calls: List[Dict[str, Any]]) -> None:
    """在界面中渲染本次调用的工具列表"""
    if not tool_calls:
        return

    st.markdown('<div class="tool-call-box">', unsafe_allow_html=True)
    st.markdown("**🛠️ 本次调用的工具：**")
    for tc in tool_calls:
        tool_name = tc.get("name", "未知工具")
        count = tc.get("count", 1)
        line = f"- <span class='tool-name'>{tool_name}</span>"
        if count > 1:
            line += f" (调用 {count} 次)"
        st.markdown(line, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_message(message: Dict[str, Any]) -> None:
    """渲染单条消息及其工具调用信息"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "tool_calls" in message:
            st.markdown('<div class="tool-call-box">', unsafe_allow_html=True)
            st.markdown("**🛠️ 使用的工具：**")
            for tool_call in message["tool_calls"]:
                tool_name = tool_call.get("name", "未知工具")
                st.markdown(
                    f"- <span class='tool-name'>{tool_name}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


def _safe_session_id(session_id: str) -> str:
    """将 session_id 规范化为安全的文件名"""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)


def get_session_log_path(session_id: str) -> Path:
    """获取 session_id 对应的日志文件路径"""
    CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_session_id(session_id or "default")
    return CHAT_LOG_DIR / f"{safe_id}.json"


def load_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """从本地 JSON 文件加载指定会话的历史消息"""
    path = get_session_log_path(session_id)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # 简单校验结构
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
    except Exception:
        # 读文件失败时忽略本地记录，避免影响正常使用
        return []
    return []


def save_session_messages(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """将当前会话消息持久化到本地 JSON 文件"""
    path = get_session_log_path(session_id)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        # 写入失败时静默处理，避免打断用户对话
        pass


