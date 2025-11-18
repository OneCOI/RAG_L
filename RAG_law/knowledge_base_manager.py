"""知识库管理模块 - 支持多知识库的创建、选择、文档导入等"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import streamlit as st

from config import (
    KNOWLEDGE_BASES_DIR,
    FAISS_INDICES_DIR,
    KB_MANAGER_CONFIG_FILE,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)


def init_directories():
    """初始化知识库和索引目录"""
    KNOWLEDGE_BASES_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDICES_DIR.mkdir(parents=True, exist_ok=True)


def load_kb_config() -> Dict:
    """加载知识库管理配置"""
    init_directories()
    
    if not KB_MANAGER_CONFIG_FILE.exists():
        # 初始化配置：兼容旧版本，将旧的知识库迁移到新结构
        config = {
            "current_kb": "default",
            "knowledge_bases": {},
        }
        
        # 检查是否存在旧的知识库目录
        old_kb_path = Path(__file__).resolve().parent / "knowledge_base"
        if old_kb_path.exists():
            # 迁移旧知识库到新结构
            default_kb_path = KNOWLEDGE_BASES_DIR / "default"
            if not default_kb_path.exists():
                import shutil
                shutil.copytree(old_kb_path, default_kb_path)
                config["knowledge_bases"]["default"] = {
                    "name": "default",
                    "path": str(default_kb_path.relative_to(Path(__file__).resolve().parent)),
                    "description": "默认知识库（从旧版本迁移）",
                    "created_at": datetime.now().isoformat(),
                }
        
        save_kb_config(config)
        return config
    
    try:
        with open(KB_MANAGER_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载知识库配置失败: {e}")
        return {"current_kb": "default", "knowledge_bases": {}}


def save_kb_config(config: Dict):
    """保存知识库管理配置"""
    try:
        with open(KB_MANAGER_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存知识库配置失败: {e}")


def get_all_knowledge_bases() -> List[str]:
    """获取所有知识库名称列表"""
    config = load_kb_config()
    return list(config.get("knowledge_bases", {}).keys())


def get_current_knowledge_base() -> str:
    """获取当前选择的知识库名称"""
    config = load_kb_config()
    return config.get("current_kb", "default")


def set_current_knowledge_base(kb_name: str):
    """设置当前选择的知识库"""
    config = load_kb_config()
    if kb_name in config.get("knowledge_bases", {}):
        config["current_kb"] = kb_name
        save_kb_config(config)


def get_knowledge_base_path(kb_name: str) -> Path:
    """获取指定知识库的路径"""
    config = load_kb_config()
    kb_info = config.get("knowledge_bases", {}).get(kb_name)
    
    if kb_info:
        # 使用配置中存储的相对路径
        base_dir = Path(__file__).resolve().parent
        return base_dir / kb_info["path"]
    else:
        # 默认路径结构
        return KNOWLEDGE_BASES_DIR / kb_name


def get_knowledge_base_index_path(kb_name: str) -> Path:
    """获取指定知识库的向量索引路径"""
    return FAISS_INDICES_DIR / kb_name


def create_knowledge_base(kb_name: str, description: str = "") -> Tuple[bool, str]:
    """
    创建新知识库

    Returns:
        (是否成功, 错误信息)
    """
    if not kb_name or not kb_name.strip():
        return False, "知识库名称不能为空"
    
    kb_name = kb_name.strip()
    
    # 检查名称合法性（不能包含特殊字符）
    if any(c in kb_name for c in r'<>:"/\|?*'):
        return False, "知识库名称不能包含特殊字符: < > : \" / \\ | ? *"
    
    config = load_kb_config()
    
    if kb_name in config.get("knowledge_bases", {}):
        return False, f"知识库 '{kb_name}' 已存在"
    
    # 创建知识库目录
    kb_path = KNOWLEDGE_BASES_DIR / kb_name
    kb_path.mkdir(parents=True, exist_ok=True)
    
    # 创建索引目录
    index_path = FAISS_INDICES_DIR / kb_name
    index_path.mkdir(parents=True, exist_ok=True)
    
    # 更新配置
    if "knowledge_bases" not in config:
        config["knowledge_bases"] = {}
    
    config["knowledge_bases"][kb_name] = {
        "name": kb_name,
        "path": str(kb_path.relative_to(Path(__file__).resolve().parent)),
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    
    save_kb_config(config)
    return True, ""


def delete_knowledge_base(kb_name: str) -> Tuple[bool, str]:
    """
    删除知识库（包括目录和索引）

    Returns:
        (是否成功, 错误信息)
    """
    config = load_kb_config()
    
    if kb_name not in config.get("knowledge_bases", {}):
        return False, f"知识库 '{kb_name}' 不存在"
    
    if kb_name == "default":
        return False, "不能删除默认知识库"
    
    # 删除目录
    kb_path = get_knowledge_base_path(kb_name)
    if kb_path.exists():
        import shutil
        shutil.rmtree(kb_path)
    
    # 删除索引
    index_path = get_knowledge_base_index_path(kb_name)
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)
    
    # 更新配置
    del config["knowledge_bases"][kb_name]
    
    # 如果删除的是当前知识库，切换到default
    if config.get("current_kb") == kb_name:
        config["current_kb"] = "default"
    
    save_kb_config(config)
    return True, ""


def get_knowledge_base_info(kb_name: str) -> Dict:
    """获取知识库信息"""
    config = load_kb_config()
    return config.get("knowledge_bases", {}).get(kb_name, {})


def get_knowledge_base_documents(kb_name: str) -> List[Path]:
    """获取知识库中的所有文档文件"""
    kb_path = get_knowledge_base_path(kb_name)
    if not kb_path.exists():
        return []
    
    documents = []
    for ext in SUPPORTED_DOCUMENT_EXTENSIONS:
        documents.extend(kb_path.glob(f"*{ext}"))
        documents.extend(kb_path.glob(f"*{ext.upper()}"))
    
    return sorted(documents)


def get_knowledge_base_stats(kb_name: str) -> Dict:
    """获取知识库统计信息"""
    docs = get_knowledge_base_documents(kb_name)
    index_path = get_knowledge_base_index_path(kb_name)
    index_file = index_path / "index.faiss"
    
    stats = {
        "document_count": len(docs),
        "total_size_mb": sum(f.stat().st_size for f in docs) / (1024 * 1024) if docs else 0,
        "has_index": index_file.exists(),
        "index_size_mb": index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0,
    }
    
    return stats

