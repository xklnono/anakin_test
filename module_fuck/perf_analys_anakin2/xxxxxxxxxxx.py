#!/usr/bin/python
# -*- coding=utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Doing trigger: ./net_exec_test_yolo and get cmd(top) info

Authors: sysqa(sysqa@baidu.com)
Date:    2018/04/09
"""

import os
import re
import sys
import time
import json
import logging
import commands
import ConfigParser

import mylogging

GLOBAL_TIME_INTERVAL = 1

def test(pain):
    print pain

if __name__ == '__main__':
    # init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    asdf = "123123"
    gpu_card = "k1200"
  
    anakin2_time_file = asdf + "/" + "Anakin2_time_%s.txt" % gpu_card
    print anakin2_time_file

    test("zxcvxzcv_%s_asdfqwer" % 12341234)
 
    create_table_sql = """CREATE TABLE IF NOT EXISTS anakin2_yolo_time_satistic_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    image_num int(6) not null,
                                    total_time float not null,
                                    average_time float not null
                                    )""" % gpu_card
    print create_table_sql

    cf = ConfigParser.ConfigParser() 
    cf.read("../conf/load_config.conf")
    conf_name = "conf_yolo"
    cmd = cf.get(conf_name, "tensorrt_ps_cmd") % gpu_card
    print cmd
