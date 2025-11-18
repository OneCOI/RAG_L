"""法律知识库管理模块（使用 RAG 组件）- 支持多知识库"""

from typing import Tuple, Optional

import streamlit as st

from components.document_loader import load_documents
from components.text_splitter import split_documents
from components.embedding import get_embedding_model
from components.vector_store import (
    load_vector_store,
    create_vector_store,
    check_index_exists as _check_index_exists,
)
from components.retriever import create_retriever
from knowledge_base_manager import (
    get_knowledge_base_path,
    get_knowledge_base_index_path,
    get_knowledge_base_stats,
)


def check_index_exists(kb_name: str) -> Tuple[bool, float]:
    """检查指定知识库的 FAISS 索引是否存在，并返回是否存在及索引大小（MB）"""
    index_path = get_knowledge_base_index_path(kb_name)
    return _check_index_exists(index_path)


@st.cache_resource
def load_legal_knowledge_base(kb_name: str, rebuild_index: bool = False):
    """
    加载/构建指定知识库（持久化 FAISS）

    这是一个高阶函数，整合了 RAG 的各个组件：
    1. 文档加载 (document_loader)
    2. 文本拆分 (text_splitter)
    3. 嵌入模型 (embedding)
    4. 向量存储 (vector_store)
    5. 检索器 (retriever)

    Args:
        kb_name: 知识库名称
        rebuild_index: 是否重建索引
    """
    kb_path = get_knowledge_base_path(kb_name)
    index_path = get_knowledge_base_index_path(kb_name)

    if not kb_path.exists():
        st.error(f"❌ 知识库目录不存在: {kb_path}")
        return None, None

    # 获取嵌入模型
    embedding_model = get_embedding_model()

    # 优先加载已有索引
    if not rebuild_index:
        db = load_vector_store(embedding_model, index_path)
        if db is not None:
            try:
                with st.spinner("📂 正在加载持久化的向量数据库..."):
                    retriever = create_retriever(db)
                    st.success("✅ 已加载持久化向量数据库")
                    return retriever, db
            except Exception as e:
                st.warning(f"⚠️ 加载持久化索引失败: {e}，将重新构建索引...")

    # 重新构建索引
    with st.spinner("🔄 正在构建向量数据库，这可能需要几分钟..."):
        try:
            # 1. 加载文档
            docs = load_documents(kb_path)

            # 2. 拆分文档（使用配置中的参数）
            split_docs = split_documents(docs)

            # 3. 创建向量存储
            db = create_vector_store(split_docs, embedding_model, index_path)

            # 4. 统计信息
            stats = get_knowledge_base_stats(kb_name)
            st.success(
                f"✅ 已构建并保存向量数据库：{stats['document_count']} 个文档，{len(split_docs)} 个文档块"
            )

            # 5. 创建检索器
            retriever = create_retriever(db)
            return retriever, db

        except Exception as e:
            st.error(f"❌ 构建知识库失败: {e}")
            return None, None
