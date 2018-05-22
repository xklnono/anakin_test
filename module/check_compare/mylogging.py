#!/usr/bin/env python
#-*-coding=utf-8-*-

################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
This module provide log file management environment.

Author   :  xukailu(xukailu@baidu.com)
Date     :  2018/04/02
"""

import os
import logging
import logging.handlers

def init_log(level=logging.DEBUG, log_path='./logs/access.log'):
    """
    mylogging - log record for different level.

    This class is used to record log and write different levels of
    information into the corresponding file. The levels are debug,
    info, warning, error, critical.

    Attributes:
                Log_Level       - msg above the level will be displayed
                                DEBUG < INFO < WARNING < ERROR < CRITICAL
                                the default value is logging.DEBUG
    """
    #找当前模块脚本真实的绝对路径
    mylogging_real_path = os.path.dirname(os.path.realpath(__file__))
    #日志存放路径
    log_path_name_access = os.path.join(mylogging_real_path, log_path)
    
    #创建一个logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    #创建一个handler，用于写入日志文件
    fh_access = logging.handlers.TimedRotatingFileHandler(log_path_name_access, 'D', 1, 10)
    
    # 定义handler的输出格式formatter    
    formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d '
                                  '* %(message)s')

    fh_access.setLevel(level)
    fh_access.setFormatter(formatter)  
    
    # 给logger添加handler    
    logger.addHandler(fh_access)  
    
    return logger

