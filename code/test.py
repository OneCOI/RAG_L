from langchain_community.tools import TavilySearchResults
import os
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
import streamlit as st
from pathlib import Path
import glob

dotenv.load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY1')

# 页面配置
st.set_page_config(
    page_title="法律 RAG 智能助手",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
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
""", unsafe_allow_html=True)

# 向量数据库持久化路径（需要在这里定义，以便在侧边栏中使用）
FAISS_INDEX_PATH = Path("code/faiss_legal_index")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # RAG 开关
    st.subheader("🔧 RAG 设置")
    use_rag = st.checkbox(
        "启用 RAG（法律知识库检索）",
        value=True,
        help="启用后可以使用法律知识库进行检索，禁用后只能使用网络搜索"
    )
    
    # 索引管理
    if use_rag:
        st.divider()
        st.subheader("📚 向量数据库管理")
        
        # 检查索引是否存在
        index_file = FAISS_INDEX_PATH / "index.faiss"
        index_pkl = FAISS_INDEX_PATH / "index.pkl"
        index_exists = index_file.exists() and index_pkl.exists()
        
        if index_exists:
            st.success("✅ 持久化索引已存在")
            index_size = index_file.stat().st_size / (1024 * 1024)  # MB
            st.caption(f"索引大小: {index_size:.2f} MB")
        else:
            st.warning("⚠️ 持久化索引不存在，将在首次使用时构建")
        
        # 重建索引按钮
        if st.button("🔄 重建向量数据库索引", use_container_width=True):
            st.session_state.rebuild_index = True
            # 清除所有相关缓存
            load_legal_knowledge_base.clear()
            initialize_rag_system.clear()
            # 清除缓存的 agent
            for key in list(st.session_state.keys()):
                if key.startswith("rag_") or key == "last_rag_setting" or key == "rag_cache_key":
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
    
    # 会话ID配置
    st.subheader("📝 会话管理")
    session_id = st.text_input(
        "会话 ID",
        value=st.session_state.get("current_session_id", "default"),
        help="不同的会话ID对应不同的对话历史",
        key="session_id_input"
    )
    
    # 更新当前会话ID
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = "default"
    
    if session_id != st.session_state.current_session_id:
        # 保存当前会话的消息
        if "messages" in st.session_state:
            if "session_messages" not in st.session_state:
                st.session_state.session_messages = {}
            st.session_state.session_messages[st.session_state.current_session_id] = st.session_state.messages.copy()
        
        # 更新会话ID
        st.session_state.current_session_id = session_id
        
        # 加载新会话的消息
        if "session_messages" in st.session_state and session_id in st.session_state.session_messages:
            st.session_state.messages = st.session_state.session_messages[session_id]
        else:
            st.session_state.messages = []
    
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
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总消息", total_messages)
        with col2:
            st.metric("用户消息", user_messages)
        st.metric("助手回复", assistant_messages)
    
    st.divider()
    
    # 关于信息
    st.subheader("ℹ️ 关于")
    st.info("""
    **法律 RAG 智能助手**
    
    这是一个基于 RAG (检索增强生成) 的法律智能助手。
    
    **功能特性:**
    - 🔍 Tavily 网络搜索
    - 📚 法律知识库检索
    - 🔧 RAG 开关控制
    - 🛠️ 工具调用可视化
    - 💬 对话历史管理
    
    **技术栈:**
    - LangChain
    - FAISS 向量数据库
    - Ollama Embeddings
    - Streamlit
    """)

# 主标题
st.markdown('<div class="main-header">⚖️ 法律 RAG 智能助手</div>', unsafe_allow_html=True)

# 加载法律知识库
@st.cache_resource
def load_legal_knowledge_base(rebuild_index=False):
    """加载法律知识库，支持持久化存储"""
    knowledge_base_path = Path("code/knowledge_base")
    if not knowledge_base_path.exists():
        st.error(f"❌ 知识库目录不存在: {knowledge_base_path}")
        return None, None
    
    # 创建嵌入模型
    embedding_model = OllamaEmbeddings(model="bge-m3:latest")
    
    # 检查是否存在持久化的向量数据库
    index_file = FAISS_INDEX_PATH / "index.faiss"
    index_pkl = FAISS_INDEX_PATH / "index.pkl"
    
    # 如果不需要重建索引且持久化文件存在，直接加载
    if not rebuild_index and index_file.exists() and index_pkl.exists():
        try:
            with st.spinner("📂 正在加载持久化的向量数据库..."):
                db = FAISS.load_local(
                    str(FAISS_INDEX_PATH),
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                retriever = db.as_retriever()
                st.success(f"✅ 成功加载持久化的向量数据库")
                return retriever, db
        except Exception as e:
            st.warning(f"⚠️ 加载持久化索引失败: {str(e)}，将重新构建索引...")
    
    # 如果不存在持久化文件或需要重建，则创建新的索引
    with st.spinner("🔄 正在构建向量数据库，这可能需要几分钟..."):
        # 获取所有 .txt 文件
        txt_files = list(knowledge_base_path.glob("*.txt"))
        if not txt_files:
            st.error(f"❌ 知识库目录中没有找到 .txt 文件")
            return None, None
        
        # 加载所有文档
        docs = []
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs.extend(loader.load())
            except Exception as e:
                st.warning(f"⚠️ 加载文件 {txt_file.name} 时出错: {str(e)}")
        
        if not docs:
            st.error("❌ 没有成功加载任何文档")
            return None, None
        
        # 拆分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(docs)
        
        # 创建向量数据库
        db = FAISS.from_documents(
            documents=split_docs,
            embedding=embedding_model
        )
        
        # 保存到本地
        try:
            FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
            db.save_local(str(FAISS_INDEX_PATH))
            st.success(f"✅ 成功构建并保存向量数据库：{len(txt_files)} 个法律文档，共 {len(split_docs)} 个文档块")
        except Exception as e:
            st.warning(f"⚠️ 保存向量数据库失败: {str(e)}")
        
        # 创建检索器
        retriever = db.as_retriever()
        
        return retriever, db

# 初始化 RAG 系统
@st.cache_resource
def initialize_rag_system(_use_rag, rebuild_index=False):
    """初始化 RAG 系统"""
    try:
        # 步骤1: 创建 Tavily 搜索工具
        search = TavilySearchResults(max_results=3)
        
        tools = [search]
        
        # 步骤2: 如果启用 RAG，加载法律知识库
        retriever = None
        if _use_rag:
            retriever, db = load_legal_knowledge_base(rebuild_index=rebuild_index)
            if retriever is not None:
                # 创建检索工具
                retriever_tool = create_retriever_tool(
                    retriever=retriever,
                    name='legal_knowledge_base',
                    description='用于检索中华人民共和国法律条文的工具，可以查询各类法律的具体内容和条款'
                )
                tools.append(retriever_tool)
        
        # 步骤3: 创建大模型
        model = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-plus"
        )
        
        # 步骤4: 创建 Agent
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(model, tools, prompt)
        
        # 步骤5: 创建 AgentExecutor（启用详细输出以追踪工具调用）
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
        
        # 步骤6: 创建带历史记录的 Agent
        store = {}
        
        def get_session_history(session_id: str):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
        
        agent_with_chat_history = RunnableWithMessageHistory(
            runnable=agent_executor,
            get_session_history=get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
        )
        
        return agent_with_chat_history, store, tools
        
    except Exception as e:
        st.error(f"❌ 初始化失败: {str(e)}")
        return None, None, []

# 初始化系统
rebuild_index = st.session_state.get("rebuild_index", False)
cache_key = f"rag_{use_rag}_rebuild_{rebuild_index}"

# 如果需要重建索引，清除缓存
if rebuild_index:
    load_legal_knowledge_base.clear()
    initialize_rag_system.clear()
    st.session_state.rebuild_index = False

if cache_key not in st.session_state or st.session_state.get("last_rag_setting") != use_rag or rebuild_index:
    agent_with_chat_history, store, tools = initialize_rag_system(use_rag, rebuild_index=rebuild_index)
    if agent_with_chat_history is not None:
        st.session_state[cache_key] = agent_with_chat_history
        st.session_state.store = store
        st.session_state.tools = tools
        st.session_state.last_rag_setting = use_rag
        st.session_state["rag_cache_key"] = cache_key
        if use_rag:
            if rebuild_index:
                st.success("✅ 系统初始化完成！（RAG 已启用，索引已重建）")
            else:
                st.success("✅ 系统初始化完成！（RAG 已启用）")
        else:
            st.success("✅ 系统初始化完成！（RAG 已禁用）")
    else:
        st.stop()
else:
    agent_with_chat_history = st.session_state[cache_key]
    tools = st.session_state.get("tools", [])

# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 欢迎消息
    rag_status = "已启用" if use_rag else "已禁用"
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"👋 您好！我是法律 RAG 智能助手。我可以帮您：\n\n1. 🔍 **网络搜索** - 回答实时问题\n2. 📚 **法律检索** - 查询中华人民共和国法律条文（RAG {rag_status}）\n3. 🛠️ **工具调用可视化** - 显示我使用的工具\n4. 💬 **对话交流** - 记住我们的对话历史\n\n请告诉我您需要什么帮助？"
    })

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 如果是助手消息且包含工具调用信息，显示工具调用
        if message["role"] == "assistant" and "tool_calls" in message:
            st.markdown('<div class="tool-call-box">', unsafe_allow_html=True)
            st.markdown("**🛠️ 使用的工具：**")
            for tool_call in message["tool_calls"]:
                tool_name = tool_call.get("name", "未知工具")
                st.markdown(f"- <span class='tool-name'>{tool_name}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# 用户输入
if user_input := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 生成助手回复
    with st.chat_message("assistant"):
        with st.spinner("🤔 正在思考中，请稍候..."):
            try:
                # 获取当前会话ID
                current_session_id = st.session_state.get('current_session_id', 'default')
                
                # 调用带历史记录的 Agent
                config = RunnableConfig(
                    configurable={'session_id': current_session_id}
                )
                
                response = agent_with_chat_history.invoke(
                    {'input': user_input},
                    config=config
                )
                
                # 提取回复内容
                response_text = response.get("output", str(response))
                
                # 提取工具调用信息
                tool_calls = []
                # 尝试从 intermediate_steps 中提取工具调用
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        if isinstance(step, (list, tuple)) and len(step) >= 1:
                            # step[0] 通常是 AgentAction 或 ToolMessage
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
                                # 检查是否已存在，如果存在则增加计数
                                existing_tool = next((tc for tc in tool_calls if tc.get("name") == tool_name), None)
                                if existing_tool:
                                    existing_tool["count"] = existing_tool.get("count", 1) + 1
                                else:
                                    tool_calls.append({"name": tool_name, "count": 1})
                
                # 显示工具调用信息
                if tool_calls:
                    st.markdown('<div class="tool-call-box">', unsafe_allow_html=True)
                    st.markdown("**🛠️ 本次调用的工具：**")
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "未知工具")
                        count = tool_call.get("count", 1)
                        tool_display = f"- <span class='tool-name'>{tool_name}</span>"
                        if count > 1:
                            tool_display += f" (调用 {count} 次)"
                        st.markdown(tool_display, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 显示回复
                st.markdown(response_text)
                
                # 保存助手回复（包含工具调用信息）
                message_to_save = {
                    "role": "assistant",
                    "content": response_text
                }
                if tool_calls:
                    message_to_save["tool_calls"] = tool_calls
                
                st.session_state.messages.append(message_to_save)
                
            except Exception as e:
                error_message = f"❌ 抱歉，处理您的问题时出现了错误：\n\n```\n{str(e)}\n```"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # 自动滚动到底部
    st.rerun()

# 底部信息
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
    unsafe_allow_html=True
)