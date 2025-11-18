from langchain_community.tools import TavilySearchResults
import os
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

# 加载环境变量
dotenv.load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY1')

# 页面配置
st.set_page_config(
    page_title="RAG 智能助手",
    page_icon="🤖",
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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .chat-container {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # 系统状态
    st.subheader("系统状态")
    if "agent_with_chat_history" not in st.session_state:
        st.warning("⏳ 系统未初始化")
    else:
        st.success("✅ RAG 系统已就绪")
    
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
    **RAG 智能助手**
    
    这是一个基于 RAG (检索增强生成) 的智能助手。
    
    **功能特性:**
    - 🔍 Tavily 网络搜索
    - 📚 暗光增强论文检索
    - 💬 对话历史管理
    
    **技术栈:**
    - LangChain
    - FAISS 向量数据库
    - Ollama Embeddings
    - Streamlit
    """)

# 主标题
st.markdown('<div class="main-header">🤖 RAG 智能助手</div>', unsafe_allow_html=True)

# 初始化 RAG 系统
@st.cache_resource
def initialize_rag_system():
    """初始化 RAG 系统，使用缓存避免重复初始化"""
    with st.spinner("正在初始化 RAG 系统，请稍候..."):
        try:
            # 步骤1: 创建 Tavily 搜索工具
            search = TavilySearchResults(max_results=3)
            
            # 步骤2: 加载文档
            loader = WebBaseLoader('https://www.cuiliangblog.cn/detail/section/234349148')
            docs = loader.load()
            
            # 步骤3: 拆分文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)
            
            # 步骤4: 创建嵌入模型和向量数据库
            embedding_model = OllamaEmbeddings(model="bge-m3:latest")
            db = FAISS.from_documents(
                documents=split_docs,
                embedding=embedding_model
            )
            
            # 步骤5: 创建检索器
            retriever = db.as_retriever()
            
            # 步骤6: 创建检索工具
            retriever_tool = create_retriever_tool(
                retriever=retriever,
                name='web_search',
                description='暗光增强论文'
            )
            
            # 步骤7: 组合工具
            tools = [search, retriever_tool]
            
            # 步骤8: 创建大模型
            model = ChatOpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model="qwen-plus"
            )
            
            # 步骤9: 创建 Agent
            prompt = hub.pull("hwchase17/openai-functions-agent")
            agent = create_tool_calling_agent(model, tools, prompt)
            
            # 步骤10: 创建 AgentExecutor
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
            
            # 步骤11: 创建带历史记录的 Agent
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
            
            return agent_with_chat_history, store
            
        except Exception as e:
            st.error(f"❌ 初始化失败: {str(e)}")
            return None, None

# 初始化系统
if "agent_with_chat_history" not in st.session_state:
    agent_with_chat_history, store = initialize_rag_system()
    if agent_with_chat_history is not None:
        st.session_state.agent_with_chat_history = agent_with_chat_history
        st.session_state.store = store
        st.success("✅ 系统初始化完成！")
    else:
        st.stop()

# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 欢迎消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 您好！我是 RAG 智能助手。我可以帮您：\n\n1. 🔍 **网络搜索** - 回答实时问题\n2. 📚 **论文检索** - 查询暗光增强相关论文内容\n3. 💬 **对话交流** - 记住我们的对话历史\n\n请告诉我您需要什么帮助？"
    })

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                response = st.session_state.agent_with_chat_history.invoke(
                    {'input': user_input},
                    config={'configurable': {'session_id': current_session_id}}
                )
                
                # 提取回复内容
                response_text = response.get("output", str(response))
                
                # 显示回复
                st.markdown(response_text)
                
                # 保存助手回复
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_message = f"❌ 抱歉，处理您的问题时出现了错误：\n\n```\n{str(e)}\n```"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # 自动滚动到底部
    st.rerun()

# 底部信息
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>
            💡 提示：您可以询问关于暗光增强的问题，或使用网络搜索功能查询实时信息
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
