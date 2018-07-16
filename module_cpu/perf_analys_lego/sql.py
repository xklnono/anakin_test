#!/usr/bin/python
# -*- coding=utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Compare paddle Main File!

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

    
    mysql = LoadCommon("cnn_seg_8")
    #mysql.create_table_sql_top_avg_model_paddle()
    #mysql.create_table_sql_nvidia_version_model_paddle()
    mysql.create_database()
    #mysql.create_table_sql_top_avg_model_paddle()
    #mysql.create_table_sql_nvidia_version_model_paddle()

#    mysql.truncate_table_sql("top_list_1sec_avg_paddle")
#    mysql.truncate_table_sql("nvidia_list_1sec_version_paddle")
