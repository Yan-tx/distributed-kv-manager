import json
import os
from types import SimpleNamespace

def load_config_from_json(config_path: str = None):
    """
    从JSON配置文件加载配置
    
    Args:
        config_path (str): 配置文件路径，默认为项目根目录的config.json
        
    Returns:
        SimpleNamespace: 包含配置信息的对象
    """
    if config_path is None:
        # 默认使用项目根目录的config.json
        # 获取当前文件所在目录的上两级目录（distributed_kv_manager的父目录）
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(project_root, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # 将字典转换为SimpleNamespace对象，支持点号访问
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    return dict_to_namespace(config_dict)

def merge_config_with_defaults(config, defaults):
    """
    将配置与默认值合并
    
    Args:
        config (SimpleNamespace): 用户配置
        defaults (dict): 默认配置字典
        
    Returns:
        SimpleNamespace: 合并后的配置
    """
    def merge_dict_with_namespace(default_dict, namespace):
        if not isinstance(namespace, SimpleNamespace):
            return namespace
            
        result = default_dict.copy()
        for key, value in vars(namespace).items():
            if key in result and isinstance(result[key], dict) and isinstance(value, SimpleNamespace):
                result[key] = merge_dict_with_namespace(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = merge_dict_with_namespace(defaults, config)
    return dict_to_namespace(merged_dict)