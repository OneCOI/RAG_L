"""文本拆分器组件"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    将文档拆分成更小的块

    Args:
        documents: 原始文档列表
        chunk_size: 每个块的大小
        chunk_overlap: 块之间的重叠大小

    Returns:
        拆分后的文档块列表
    """
    if chunk_size is None:
        chunk_size = TEXT_CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = TEXT_CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

