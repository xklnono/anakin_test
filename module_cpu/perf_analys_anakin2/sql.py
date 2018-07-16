#!/usr/bin/python
# -*- coding=utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Compare TensorRT Main File!

Authors: sysqa(sysqa@baidu.com)
Date:    2018/04/04
"""

import os
import re
import sys
import time
import json
import logging
import random

import mylogging
from load_common import LoadCommon



if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    
    mysql = LoadCommon("language", 1, "5177")
    #mysql.create_table_sql_top_avg_model_tensorRT()
    #mysql.create_table_sql_nvidia_version_model_tensorRT()
    mysql.create_database()
