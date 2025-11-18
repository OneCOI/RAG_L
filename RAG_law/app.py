import streamlit as st

from langchain_core.runnables import RunnableConfig

from config import (
    ST_PAGE_TITLE,
    ST_PAGE_ICON,
    ST_LAYOUT,
    ST_INITIAL_SIDEBAR_STATE,
    DEFAULT_SESSION_ID,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from knowledge_base import check_index_exists, load_legal_knowledge_base
from knowledge_base_manager import (
    get_all_knowledge_bases,
    get_current_knowledge_base,
    set_current_knowledge_base,
    create_knowledge_base,
    delete_knowledge_base,
    get_knowledge_base_path,
    get_knowledge_base_stats,
    get_knowledge_base_documents,
)
from components.document_loader import save_uploaded_file
from rag_system import initialize_rag_system
from utils import (
    extract_tool_calls,
    render_message,
    render_tool_calls,
    load_session_messages,
    save_session_messages,
)

# 页面配置
st.set_page_config(
    page_title=ST_PAGE_TITLE,
    page_icon=ST_PAGE_ICON,
    layout=ST_LAYOUT,
    initial_sidebar_state=ST_INITIAL_SIDEBAR_STATE,
)

# 自定义 CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
        padding: 1rem;
    }
    .tool-call-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .tool-name {
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")

    # 知识库管理
    st.subheader("📚 知识库管理")
    
    # 获取所有知识库
    all_kbs = get_all_knowledge_bases()
    current_kb = get_current_knowledge_base()
    
    # 如果当前知识库不在列表中，使用默认知识库
    if current_kb not in all_kbs:
        if all_kbs:
            current_kb = all_kbs[0]
        else:
            # 创建默认知识库
            create_knowledge_base("default", "默认知识库")
            all_kbs = get_all_knowledge_bases()
            current_kb = "default"
        set_current_knowledge_base(current_kb)
    
    # 知识库选择下拉框
    selected_kb = st.selectbox(
        "选择知识库",
        options=all_kbs,
        index=all_kbs.index(current_kb) if current_kb in all_kbs else 0,
        help="选择要使用的知识库",
        key="kb_selector"
    )
    
    # 如果切换了知识库，更新当前知识库
    if selected_kb != current_kb:
        set_current_knowledge_base(selected_kb)
        # 清除缓存，重新加载
        load_legal_knowledge_base.clear()
        initialize_rag_system.clear()
        for key in list(st.session_state.keys()):
            if key.startswith("rag_") or key in {"last_rag_setting", "rag_cache_key", "last_kb_name"}:
                del st.session_state[key]
        st.rerun()
    
    # 当前知识库信息
    if selected_kb:
        stats = get_knowledge_base_stats(selected_kb)
        st.caption(f"📄 {stats['document_count']} 个文档")
        st.caption(f"💾 {stats['total_size_mb']:.2f} MB")
        if stats['has_index']:
            st.caption(f"✅ 索引: {stats['index_size_mb']:.2f} MB")
        else:
            st.caption("⚠️ 索引未构建")
    
    # 知识库操作
    st.divider()
    
    # 创建新知识库
    with st.expander("➕ 创建新知识库"):
        new_kb_name = st.text_input("知识库名称", key="new_kb_name")
        new_kb_desc = st.text_input("描述（可选）", key="new_kb_desc")
        if st.button("创建", key="create_kb_btn"):
            if new_kb_name:
                success, error = create_knowledge_base(new_kb_name.strip(), new_kb_desc.strip())
                if success:
                    st.success(f"✅ 知识库 '{new_kb_name}' 创建成功！")
                    st.rerun()
                else:
                    st.error(f"❌ {error}")
            else:
                st.warning("⚠️ 请输入知识库名称")
    
    # 上传文档
    if selected_kb:
        with st.expander("📤 上传文档"):
            uploaded_files = st.file_uploader(
                "选择文档文件",
                type=[ext.replace(".", "") for ext in SUPPORTED_DOCUMENT_EXTENSIONS],
                accept_multiple_files=True,
                help=f"支持格式: {', '.join(SUPPORTED_DOCUMENT_EXTENSIONS)}"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    save_path = get_knowledge_base_path(selected_kb) / uploaded_file.name
                    success, error = save_uploaded_file(uploaded_file, save_path)
                    if success:
                        st.success(f"✅ {uploaded_file.name} 上传成功")
                    else:
                        st.error(f"❌ {uploaded_file.name} 上传失败: {error}")
                
                # 清除缓存，需要重建索引
                if success:
                    load_legal_knowledge_base.clear()
                    st.info("💡 上传文档后，请点击下方「重建索引」按钮更新向量数据库")
    
    # 删除知识库
    if selected_kb and selected_kb != "default":
        with st.expander("🗑️ 删除知识库", expanded=False):
            st.warning(f"⚠️ 删除知识库 '{selected_kb}' 将删除所有文档和索引，此操作不可恢复！")
            if st.button("确认删除", key="delete_kb_btn", type="secondary"):
                success, error = delete_knowledge_base(selected_kb)
                if success:
                    st.success(f"✅ 知识库 '{selected_kb}' 已删除")
                    st.rerun()
                else:
                    st.error(f"❌ {error}")

    # RAG 开关
    st.divider()
    st.subheader("🔧 RAG 设置")
    use_rag = st.checkbox(
        "启用 RAG（知识库检索）",
        value=True,
        help="启用后可以使用知识库进行检索，禁用后只能使用网络搜索",
    )

    # 索引管理
    if use_rag and selected_kb:
        st.divider()
        st.subheader("🔍 向量索引管理")

        index_exists, index_size = check_index_exists(selected_kb)
        if index_exists:
            st.success("✅ 持久化索引已存在")
            st.caption(f"索引大小: {index_size:.2f} MB")
        else:
            st.warning("⚠️ 持久化索引不存在，将在首次使用时构建")

        if st.button("🔄 重建向量数据库索引", use_container_width=True):
            st.session_state.rebuild_index = True
            st.session_state.rebuild_kb_name = selected_kb
            load_legal_knowledge_base.clear()
            initialize_rag_system.clear()
            for key in list(st.session_state.keys()):
                if key.startswith("rag_") or key in {"last_rag_setting", "rag_cache_key", "last_kb_name"}:
                    del st.session_state[key]
            st.rerun()

    # 系统状态
    st.divider()
    st.subheader("系统状态")
    if "retriever_tool" not in st.session_state and use_rag:
        st.warning("⏳ 正在加载法律知识库...")
    else:
        st.success("✅ 系统已就绪")

    st.divider()

    # 会话 ID
    st.subheader("📝 会话管理")
    session_id = st.text_input(
        "会话 ID",
        value=st.session_state.get("current_session_id", DEFAULT_SESSION_ID),
        help="不同的会话ID对应不同的对话历史",
        key="session_id_input",
    )

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = DEFAULT_SESSION_ID

    if session_id != st.session_state.current_session_id:
        # 先将当前会话消息写入持久化存储
        if "messages" in st.session_state:
            save_session_messages(st.session_state.current_session_id, st.session_state.messages)

        st.session_state.current_session_id = session_id

        # 尝试从内存或磁盘加载新会话消息
        if (
            "session_messages" in st.session_state
            and session_id in st.session_state.session_messages
        ):
            st.session_state.messages = st.session_state.session_messages[session_id]
        else:
            loaded = load_session_messages(session_id)
            st.session_state.messages = loaded or []

    if st.button("🔄 刷新会话", use_container_width=True):
        st.rerun()

    if st.button("🗑️ 清空当前会话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # 统计信息
    st.divider()
    st.subheader("📊 统计信息")
    if "messages" in st.session_state:
        total_messages = len(st.session_state.messages)
        user_messages = len(
            [m for m in st.session_state.messages if m["role"] == "user"]
        )
        assistant_messages = len(
            [m for m in st.session_state.messages if m["role"] == "assistant"]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("总消息", total_messages)
        with col2:
            st.metric("用户消息", user_messages)
        st.metric("助手回复", assistant_messages)

    st.divider()

    st.subheader("ℹ️ 关于")
    st.info(
        """
    **法律 RAG 智能助手**

    - 🔍 Tavily 网络搜索
    - 📚 法律知识库检索
    - 🔧 RAG 开关控制
    - 🛠️ 工具调用可视化
    - 💬 对话历史管理
    """
    )


# 主标题
st.markdown(
    '<div class="main-header">⚖️ 法律 RAG 智能助手</div>', unsafe_allow_html=True
)


# 获取当前知识库名称（在侧边栏定义use_rag之后获取）
# 注意：这里需要确保侧边栏代码已执行完毕，use_rag 和 selected_kb 已定义
# current_kb_name 在侧边栏中已经通过 set_current_knowledge_base 设置

# 初始化系统（包含重建索引控制）
current_kb_name = get_current_knowledge_base()
rebuild_index = st.session_state.get("rebuild_index", False)
rebuild_kb_name = st.session_state.get("rebuild_kb_name", current_kb_name)
cache_key = f"rag_{use_rag}_kb_{current_kb_name}_rebuild_{rebuild_index}"

if rebuild_index:
    load_legal_knowledge_base.clear()
    initialize_rag_system.clear()
    st.session_state.rebuild_index = False
    if "rebuild_kb_name" in st.session_state:
        del st.session_state["rebuild_kb_name"]

if (
    cache_key not in st.session_state
    or st.session_state.get("last_rag_setting") != use_rag
    or st.session_state.get("last_kb_name") != current_kb_name
    or rebuild_index
):
    agent_with_history, store, tools = initialize_rag_system(
        use_rag=use_rag, kb_name=current_kb_name, rebuild_index=rebuild_index
    )
    if agent_with_history is not None:
        st.session_state[cache_key] = agent_with_history
        st.session_state.store = store
        st.session_state.tools = tools
        st.session_state["last_rag_setting"] = use_rag
        st.session_state["last_kb_name"] = current_kb_name
        st.session_state["rag_cache_key"] = cache_key
        if use_rag:
            msg = "✅ 系统初始化完成！（RAG 已启用，索引已重建）" if rebuild_index else "✅ 系统初始化完成！（RAG 已启用）"
        else:
            msg = "✅ 系统初始化完成！（RAG 已禁用）"
        st.success(msg)
    else:
        st.stop()
else:
    agent_with_history = st.session_state[cache_key]
    tools = st.session_state.get("tools", [])


# 初始化消息历史（优先从磁盘加载）
if "messages" not in st.session_state:
    current_session_id = st.session_state.get("current_session_id", DEFAULT_SESSION_ID)
    loaded = load_session_messages(current_session_id)
    if loaded:
        st.session_state.messages = loaded
    else:
        st.session_state.messages = []
        rag_status = "已启用" if use_rag else "已禁用"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "👋 您好！我是法律 RAG 智能助手。我可以帮您：\n\n"
                    "1. 🔍 **网络搜索** - 回答实时问题\n"
                    "2. 📚 **法律检索** - 查询中华人民共和国法律条文"
                    f"（RAG {rag_status}）\n"
                    "3. 🛠️ **工具调用可视化** - 显示我使用的工具\n"
                    "4. 💬 **对话交流** - 记住我们的对话历史\n\n"
                    "请告诉我您需要什么帮助？"
                ),
            }
        )
        # 保存欢迎消息
        save_session_messages(current_session_id, st.session_state.messages)


# 展示历史对话
for message in st.session_state.messages:
    render_message(message)


# 聊天输入
if user_input := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_session_messages(st.session_state.get("current_session_id", "default"), st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在思考中，请稍候..."):
            try:
                current_session_id = st.session_state.get("current_session_id", DEFAULT_SESSION_ID)
                config = RunnableConfig(configurable={"session_id": current_session_id})

                response = agent_with_history.invoke({"input": user_input}, config=config)

                response_text = response.get("output", str(response))

                tool_calls = extract_tool_calls(response)

                # 显示工具调用信息
                render_tool_calls(tool_calls)

                st.markdown(response_text)

                msg_to_save = {"role": "assistant", "content": response_text}
                if tool_calls:
                    msg_to_save["tool_calls"] = tool_calls
                st.session_state.messages.append(msg_to_save)
                # 持久化保存整个对话
                save_session_messages(
                    st.session_state.get("current_session_id", DEFAULT_SESSION_ID),
                    st.session_state.messages,
                )

            except Exception as e:
                err = f"❌ 抱歉，处理您的问题时出现了错误：\n\n```\n{e}\n```"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

    st.rerun()


st.divider()
rag_status_text = "已启用" if use_rag else "已禁用"
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>
            💡 提示：您可以询问法律相关问题，或使用网络搜索功能查询实时信息。当前 RAG 状态：<strong>{rag_status_text}</strong>
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)


