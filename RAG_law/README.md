# 法律 RAG 智能助手

基于 RAG (检索增强生成) 架构的法律智能助手系统。

## 项目结构

```
RAG_law/
├── components/           # RAG 核心组件
│   ├── document_loader.py    # 文档加载器
│   ├── text_splitter.py      # 文本拆分器
│   ├── embedding.py          # 嵌入模型
│   ├── vector_store.py       # 向量存储（FAISS）
│   ├── retriever.py          # 检索器
│   ├── llm.py                # 大语言模型
│   ├── tools.py              # 工具定义（搜索、检索工具）
│   └── agent.py              # Agent 和 AgentExecutor
│
├── config.py            # 全局配置（路径、API Key）
├── knowledge_base.py    # 知识库管理（整合各组件）
├── rag_system.py        # RAG 系统初始化（整合各组件）
├── utils.py             # 工具函数（UI渲染、对话历史管理）
├── app.py               # Streamlit 应用主入口
│
├── knowledge_base/      # 法律文档知识库（.txt 文件）
├── faiss_legal_index/   # FAISS 向量索引持久化存储
└── chat_logs/           # 对话历史持久化存储
```

## RAG 组件说明

### 1. Document Loader（文档加载器）
- **文件**: `components/document_loader.py`
- **功能**: 从 `knowledge_base/` 目录加载法律文档（.txt 文件）
- **组件**: `load_documents()`

### 2. Text Splitter（文本拆分器）
- **文件**: `components/text_splitter.py`
- **功能**: 将长文档拆分成适合向量化的文档块
- **组件**: `split_documents()`

### 3. Embedding（嵌入模型）
- **文件**: `components/embedding.py`
- **功能**: 提供文本向量化模型（Ollama bge-m3）
- **组件**: `get_embedding_model()`

### 4. Vector Store（向量存储）
- **文件**: `components/vector_store.py`
- **功能**: FAISS 向量数据库的创建、加载、持久化
- **组件**: `create_vector_store()`, `load_vector_store()`, `check_index_exists()`

### 5. Retriever（检索器）
- **文件**: `components/retriever.py`
- **功能**: 从向量存储创建检索器，用于相似度搜索
- **组件**: `create_retriever()`

### 6. LLM（大语言模型）
- **文件**: `components/llm.py`
- **功能**: 配置和初始化大语言模型（Qwen-plus）
- **组件**: `get_llm()`

### 7. Tools（工具）
- **文件**: `components/tools.py`
- **功能**: 定义 Agent 可用的工具（网络搜索、法律知识库检索）
- **组件**: `create_search_tool()`, `create_legal_retriever_tool()`, `get_tools()`

### 8. Agent（智能体）
- **文件**: `components/agent.py`
- **功能**: 创建 Agent Executor 和带历史记录的 Agent
- **组件**: `create_agent_executor()`, `create_agent_with_history()`

## 数据流

```
法律文档 (.txt)
    ↓
Document Loader（文档加载器）
    ↓
Text Splitter（文本拆分器）
    ↓
Embedding（嵌入模型）
    ↓
Vector Store（向量存储）→ FAISS 持久化
    ↓
Retriever（检索器）
    ↓
Tools（工具）
    ↓
Agent（智能体）
    ↓
LLM（大语言模型）
    ↓
用户问答
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在 `.env` 文件中配置：

```env
TAVILY_API_KEY1=your_tavily_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
```

### 3. 运行应用

```bash
streamlit run app.py
```

## 功能特性

- ✅ **RAG 组件化**: 按照标准 RAG 架构拆分组件，易于维护和扩展
- ✅ **持久化存储**: 向量索引和对话历史都支持持久化
- ✅ **工具调用可视化**: 显示每次调用使用的工具
- ✅ **多会话管理**: 支持多个会话 ID，每个会话独立对话历史
- ✅ **RAG 开关**: 可以动态启用/禁用法律知识库检索

## 组件设计原则

1. **单一职责**: 每个组件只负责一个功能
2. **依赖注入**: 组件之间通过参数传递依赖，而不是硬编码
3. **可扩展性**: 易于替换组件（如更换嵌入模型、向量存储等）
4. **可测试性**: 每个组件都可以独立测试

