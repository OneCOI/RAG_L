"""嵌入模型组件"""

from langchain_ollama import OllamaEmbeddings

from config import EMBEDDING_MODEL_NAME


def get_embedding_model(model_name: str = None) -> OllamaEmbeddings:
    """
    获取嵌入模型实例

    Args:
        model_name: Ollama 模型名称

    Returns:
        嵌入模型实例
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL_NAME
    return OllamaEmbeddings(model=model_name)

