"""文档加载器组件 - 支持多种文档格式"""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

from config import SUPPORTED_DOCUMENT_EXTENSIONS


def load_documents(knowledge_base_path: Path) -> List[Document]:
    """
    加载知识库中的所有文档（支持多种格式）

    Args:
        knowledge_base_path: 知识库目录路径

    Returns:
        文档列表
    """
    if not knowledge_base_path.exists():
        raise FileNotFoundError(f"知识库目录不存在: {knowledge_base_path}")

    # 收集所有支持的文档文件
    doc_files = []
    for ext in SUPPORTED_DOCUMENT_EXTENSIONS:
        doc_files.extend(knowledge_base_path.glob(f"*{ext}"))
        doc_files.extend(knowledge_base_path.glob(f"*{ext.upper()}"))

    if not doc_files:
        raise FileNotFoundError(f"知识库目录中没有找到支持的文档文件: {SUPPORTED_DOCUMENT_EXTENSIONS}")

    docs = []
    for doc_file in doc_files:
        try:
            loader = get_document_loader(doc_file)
            if loader:
                docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ 加载文件 {doc_file.name} 时出错: {e}")

    if not docs:
        raise ValueError("没有成功加载任何文档")

    return docs


def get_document_loader(file_path: Path) -> Optional[object]:
    """
    根据文件扩展名返回对应的文档加载器

    Args:
        file_path: 文件路径

    Returns:
        文档加载器实例，如果不支持则返回 None
    """
    ext = file_path.suffix.lower()

    if ext == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")
    elif ext == ".pdf":
        return PyPDFLoader(str(file_path))
    elif ext == ".md":
        return UnstructuredMarkdownLoader(str(file_path))
    elif ext in [".docx", ".doc"]:
        try:
            return Docx2txtLoader(str(file_path))
        except ImportError:
            raise ImportError(
                "加载 .docx/.doc 文件需要安装 docx2txt: pip install docx2txt"
            )
    else:
        return None


def save_uploaded_file(uploaded_file, save_path: Path) -> Tuple[bool, str]:
    """
    保存上传的文件到指定路径

    Args:
        uploaded_file: Streamlit UploadedFile 对象
        save_path: 保存路径

    Returns:
        (是否成功, 错误信息)
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return True, ""
    except Exception as e:
        return False, str(e)

