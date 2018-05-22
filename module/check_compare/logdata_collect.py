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

import mylogging
from load_common import LoadCommon

MAIN_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_NAME = "./logs/access.log"
#LOG_NAME = "./logs/access.log.2018-04-03"
LOG_PATH = os.path.join(MAIN_PATH, LOG_NAME)


if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    #check src_path and dst_path
    if not os.path.exists(LOG_PATH):
        print ("\033[0;31;m[error]: Pls Check The LOG File Path!\nlog file: %s\033[0m" % (LOG_PATH))
        sys.exit(1)

    total_num = 0 
    pass_num = 0
    no_pass_num = 0
    no_right_shape = 0 
    no_right_data = 0 
    no_right_weight = 0
    no_right_c_and_h = 0

    f = open(LOG_PATH)
    for line in f.readlines():
        status = line.split()[5].rstrip(":")
        if status == "NO_RIGHT_SHAPE":
            no_right_shape += 1
        elif status == "NO_RIGHT_DATA":
            no_right_data += 1
        elif status == "NO_RIGHT_WEIGHT":
            no_right_weight += 1
        elif status == "NO_RIGHT_C_AND_H":
            no_right_c_and_h += 1
        elif status == "NO_PASS":
            no_pass_num += 1
            total_num += 1
        elif status == "PASS":
            pass_num += 1
            total_num += 1

    f.close()
    print ("\033[0;36;m[total_num]        = %s\033[0m" % (total_num))
    print ("\033[0;36;m[pass_num]         = %s\033[0m" % (pass_num))
    print ("\033[0;36;m[no_pass_num]      = %s\033[0m" % (no_pass_num))
    if total_num == 0:
        print "[error] total_num==0, pls check!!!!!!!!!!"
        sys.exit(1)
    success_ratio = round(float(pass_num)/float(total_num), 4)
    print ("\033[0;36;m[success_ratio]    = %s\033[0m" % (success_ratio))

#    print ("\033[0;36;m[no_right_shape]   = (%s)\033[0m" % (no_right_shape))
#    print ("\033[0;36;m[no_right_data]    = (%s)\033[0m" % (no_right_data))
#    print ("\033[0;36;m[no_right_weight]  = (%s)\033[0m" % (no_right_weight))
#    print ("\033[0;36;m[no_right_c_and_h] = (%s)\033[0m" % (no_right_c_and_h))

    time_now = int(time.time())
    time_local = time.localtime(time_now)
    time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    if len(sys.argv) == 1:
        print ("\033[0;36;mno right input model: python logdata_collect.py \"model\"\033[0m")
        sys.exit(1)
    elif len(sys.argv) == 2:
        model = sys.argv[1]
        batch_size = 1
        gpu_card = "p4"
        mysql = LoadCommon(model, batch_size, gpu_card)
    elif len(sys.argv) == 3:
        model = sys.argv[1]
        batch_size = sys.argv[2]
        gpu_card = "p4"
        mysql = LoadCommon(model, batch_size, gpu_card)
    elif len(sys.argv) == 4:
        model = sys.argv[1]
        batch_size = sys.argv[2]
        gpu_card = sys.argv[3]
        mysql = LoadCommon(model, batch_size, gpu_card)
    elif len(sys.argv) == 5:
        model = sys.argv[1]
        batch_size = sys.argv[2]
        gpu_card = sys.argv[3]
        mysql = LoadCommon(model, batch_size, gpu_card)

        # use the same time_now for mysql data
        #time_now = int(time.time())
        time_now = int(sys.argv[4])
        time_local = time.localtime(time_now)
        time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    mysql.create_table_sql()
    mysql.insert_table_sql(time_sql, total_num, pass_num, no_pass_num, success_ratio)

    #rename logs/access.log to logs/access.log.time
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S",time_local)
    cmd = "mv ./logs/access.log ./logs/access.log.%s" % time_name
    os.system(cmd)

