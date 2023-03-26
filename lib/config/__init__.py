"""
 config:配置库
 =============

 包含了项目的所有配置信息，通过调用cfg(yacs.config.CfgNode类，包含所有显示设定的配置参数)和args(Namespcae对象，包含所有的命令行参数)来设定项目的参数    
"""
from .config import cfg, args
