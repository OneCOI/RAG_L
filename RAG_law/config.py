"""全局配置模块 - 所有可修改的参数都在这里"""

from pathlib import Path
import os
import dotenv

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录（RAG_law）
BASE_DIR = Path(__file__).resolve().parent

# 知识库管理目录
KNOWLEDGE_BASES_DIR = BASE_DIR / "knowledge_bases"  # 多个知识库的根目录
FAISS_INDICES_DIR = BASE_DIR / "faiss_indices"  # 多个向量索引的根目录

# 对话日志持久化目录
CHAT_LOG_DIR = BASE_DIR / "chat_logs"

# 知识库管理配置文件
KB_MANAGER_CONFIG_FILE = BASE_DIR / "kb_manager_config.json"

# 支持的文档格式
SUPPORTED_DOCUMENT_EXTENSIONS = [".txt", ".pdf", ".md", ".docx", ".doc"]

# ============================================================================
# 环境变量 & API Key
# ============================================================================

dotenv.load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# ============================================================================
# 文档处理配置
# ============================================================================

# 文本拆分参数
TEXT_CHUNK_SIZE = 50  # 每个文档块的大小（字符数）
TEXT_CHUNK_OVERLAP = 15  # 文档块之间的重叠大小（字符数）

# 注意：法律条文通常较短，使用较小的 chunk_size 有利于精细匹配
# 如果处理长文档，可以调整为：chunk_size=1000, chunk_overlap=100

# ============================================================================
# 嵌入模型配置
# ============================================================================

# Ollama 嵌入模型名称
EMBEDDING_MODEL_NAME = "bge-m3:latest"

# 可选的其他模型：
# - "bge-large-zh-v1.5"
# - "text-embedding-ada-002" (需要 OpenAI API)

# ============================================================================
# 大语言模型配置
# ============================================================================

# LLM 模型名称
LLM_MODEL_NAME = "qwen-plus"

# LLM API 基础 URL
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LLM 其他参数（可选）
LLM_TEMPERATURE = None  # 默认值，可设置为 0.0-2.0
LLM_MAX_TOKENS = None  # 默认值，可设置为整数

# 可选的其他模型配置：
# - OpenAI: base_url="https://api.openai.com/v1", model="gpt-4"
# - 本地模型: base_url="http://localhost:11434/v1", model="llama2"

# ============================================================================
# 工具配置
# ============================================================================

# Tavily 网络搜索工具
TAVILY_MAX_RESULTS = 3  # 最大搜索结果数

# 法律知识库检索工具
LEGAL_TOOL_NAME = "legal_knowledge_base"
LEGAL_TOOL_DESCRIPTION = (
    "用于检索中华人民共和国法律条文的工具，可以查询各类法律的具体内容和条款"
)

# ============================================================================
# Agent 配置
# ============================================================================

# Agent Executor 参数
AGENT_VERBOSE = True  # 是否输出详细信息（用于调试）
AGENT_RETURN_INTERMEDIATE_STEPS = True  # 是否返回中间步骤（用于工具调用可视化）

# Agent Prompt（从 LangChain Hub 拉取）
AGENT_PROMPT_HUB = "hwchase17/openai-functions-agent"

# ============================================================================
# Streamlit UI 配置
# ============================================================================

# 页面配置
ST_PAGE_TITLE = "法律 RAG 智能助手"
ST_PAGE_ICON = "⚖️"
ST_LAYOUT = "wide"  # "wide" 或 "centered"
ST_INITIAL_SIDEBAR_STATE = "expanded"  # "expanded" 或 "collapsed"

# 默认会话 ID
DEFAULT_SESSION_ID = "default"

# ============================================================================
# 检索器配置
# ============================================================================

# 检索器参数（可选，传递给 as_retriever()）
RETRIEVER_SEARCH_KWARGS = None  # 例如: {"k": 5} 表示返回 top-5 结果
RETRIEVER_SEARCH_TYPE = "similarity"  # "similarity" 或 "mmr"

# ============================================================================
# 向量存储配置
# ============================================================================

# FAISS 索引文件名
FAISS_INDEX_FILE = "index.faiss"
FAISS_INDEX_PKL = "index.pkl"

# ============================================================================
# 对话历史配置
# ============================================================================

# 对话日志文件格式
CHAT_LOG_FILE_EXT = ".json"  # 对话历史保存为 JSON 格式

# ============================================================================
# 功能开关
# ============================================================================

# 默认是否启用 RAG
DEFAULT_USE_RAG = True

# 是否启用对话历史持久化
ENABLE_CHAT_PERSISTENCE = True
