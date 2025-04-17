'''
Author: Shukun
Date: 2025-03-26 17:03:28
LastEditors: Shukun
LastEditTime: 2025-04-01 20:40:58
Description: 初始化自身环境的初始条件
'''
import argparse
import os
import sys
from xuance.common import get_configs
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 返回上一级目录，即项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)
configs_dict = get_configs(file_dir="config\ConflictResolution.yaml")
configs = argparse.Namespace(**configs_dict)
# 如果 config.args 是一个 dict，就手动转换一次
if isinstance(configs.args, dict):
    configs.args = argparse.Namespace(**configs.args)
from config.ConflictResolutionEnv import MyNewMultiAgentEnv
from xuance.environment import REGISTRY_MULTI_AGENT_ENV
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv
