"""向量存储组件（FAISS）"""

from pathlib import Path
from typing import Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# 注意：不再使用全局 FAISS_INDEX_PATH，改为传入参数


def create_vector_store(
    documents: list[Document],
    embedding_model: OllamaEmbeddings,
    save_path: Path,
) -> FAISS:
    """
    创建向量存储

    Args:
        documents: 文档列表
        embedding_model: 嵌入模型
        save_path: 保存路径（必须提供）

    Returns:
        FAISS 向量存储实例
    """
    if save_path is None:
        raise ValueError("save_path 不能为 None")

    db = FAISS.from_documents(documents=documents, embedding=embedding_model)

    # 保存到本地
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        db.save_local(str(save_path))
    except Exception as e:
        print(f"⚠️ 保存向量数据库失败: {e}")

    return db


def load_vector_store(
    embedding_model: OllamaEmbeddings,
    load_path: Path,
) -> Optional[FAISS]:
    """
    从本地加载向量存储

    Args:
        embedding_model: 嵌入模型
        load_path: 加载路径（必须提供）

    Returns:
        FAISS 向量存储实例，如果不存在则返回 None
    """
    if load_path is None:
        raise ValueError("load_path 不能为 None")

    index_file = load_path / "index.faiss"
    index_pkl = load_path / "index.pkl"

    if not (index_file.exists() and index_pkl.exists()):
        return None

    try:
        db = FAISS.load_local(
            str(load_path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        return db
    except Exception as e:
        print(f"⚠️ 加载向量数据库失败: {e}")
        return None


def check_index_exists(index_path: Path) -> Tuple[bool, float]:
    """
    检查索引是否存在，并返回索引大小（MB）

    Args:
        index_path: 索引路径（必须提供）

    Returns:
        (是否存在, 索引大小MB)
    """
    if index_path is None:
        raise ValueError("index_path 不能为 None")

    index_file = index_path / "index.faiss"
    index_pkl = index_path / "index.pkl"
    exists = index_file.exists() and index_pkl.exists()
    size_mb = 0.0

    if exists:
        try:
            size_mb = index_file.stat().st_size / (1024 * 1024)
        except Exception:
            size_mb = 0.0

    return exists, size_mb

