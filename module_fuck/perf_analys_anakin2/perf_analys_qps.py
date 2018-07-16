#!/usr/bin/python
# -*- coding=utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Load Performance Analysis Data

Authors: sysqa(sysqa@baidu.com)
Date:    2018/04/08
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
from load_common import LoadCommon

MYSQL_DB_ONOFF = "off"
#MYSQL_DB_ONOFF = "on"


class LoadPerformance(object):
    """
    init
    """
    def __init__(self, time_sql, mysql_w, model="", batch_size=1, gpu_card="p4"):
        """
        init
        """
        try:
            cf = ConfigParser.ConfigParser()
            #cf.read("./conf/load_config.conf")
            cf.read("../conf/load_config.conf")
            conf_name = "conf_%s" % model
            try:
                self.gpu_card = gpu_card
                self.time_pk_path = cf.get(conf_name, "time_pk_path")
            except Exception as e:
                print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
                sys.exit(1)
            
            # init mysql
            self.mysql = LoadCommon(model, batch_size, self.gpu_card)

            # init the initing time
            # the qps table has the same primary key: time
            self.time_sql = time_sql

            # init env: truncate table top_list_1sec 
            self.mysql.create_table_sql_anakin2_yolo_qps()
            self.mysql.create_table_sql_tensorrt_qps()
            self.db_onoff = mysql_w
        except Exception as exception:
            print exception
            return

    def analysis_tensorRT_qps(self):
        """
        analysis the anakin1.0's qps
        """
        anakin1_time_file = self.time_pk_path + "/" + "TensorRT_time_%s.txt" % self.gpu_card
        if not os.path.exists(anakin1_time_file):
            print "[alarm]: the TensorRT_time.txt file do not exist"
            sys.exit(0)
        try:
            f = open(anakin1_time_file)
            #for line in f.readlines():
            #    if line.split()[0] == "image_num":
            #        image_num = int(line.split()[2])
            #    elif line.split()[0] == "total_time":
            #        total_time_ms = float(line.split()[2].strip("ms"))
            #        total_time = total_time_ms / 1000
            #    elif line.split()[0] == "average_time":
            #        average_time = float(line.split()[2].strip("ms"))
            for line in f.readlines():
                if "image_num" in line.split(":")[0]:
                    image_num = int(line.split(":")[1].strip())
                elif "total_time" in line.split(":")[0]:
                    total_time_ms = float(line.split(":")[1].strip().strip("ms"))
                    total_time = total_time_ms / 1000
                elif "average_time" in line.split(":")[0]:
                    average_time = float(line.split(":")[1].strip().strip("ms"))
            if self.db_onoff == "on":
                self.mysql.insert_table_sql_tensorrt_qps(self.time_sql, image_num, total_time, average_time)
        except Exception,e:
            print "\033[0;31;m[error]: analysis TensorRT_time.txt file error\033[0m"
            sys.exit(1)
        finally:
            f.close()
        

    def analysis_anakin2_qps(self):
        """
        analysis the anakin2.0's qps
        """
        anakin2_time_file = self.time_pk_path + "/" + "Anakin2_time_%s.txt" % self.gpu_card
        if not os.path.exists(anakin2_time_file):
            print "[alarm]: the Anakin2_time.txt file do not exist"
            sys.exit(0)
        try:
            f = open(anakin2_time_file)
            #for line in f.readlines():
            #    if line.split()[0] == "image_num":
            #        image_num = int(line.split()[2])
            #    elif line.split()[0] == "total_time":
            #        total_time_ms = float(line.split()[2].strip("ms"))
            #        total_time = total_time_ms / 1000
            #    elif line.split()[0] == "average_time":
            #        average_time = float(line.split()[2].strip("ms"))
            for line in f.readlines():
                if "image_num" in line.split(":")[0]:
                    image_num = int(line.split(":")[1].strip())
                elif "total_time" in line.split(":")[0]:
                    total_time_ms = float(line.split(":")[1].strip().strip("ms"))
                    total_time = total_time_ms / 1000
                elif "average_time" in line.split(":")[0]:
                    average_time = float(line.split(":")[1].strip().strip("ms"))
            if self.db_onoff == "on":
                self.mysql.insert_table_sql_anakin2_yolo_qps(self.time_sql, image_num, total_time, average_time)
        except Exception,e:
            print "\033[0;31;m[error]: analysis Anakin2_time.txt file error\033[0m"
            sys.exit(1)
        finally:
            f.close()

if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    global MYSQL_DB_ONOFF

    time_now = int(time.time())
    time_local = time.localtime(time_now)
    time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    if len(sys.argv) == 6:
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        batch_size = sys.argv[3]
        gpu_card = sys.argv[4]

        # use the same time_now for mysql data
        #time_now = int(time.time())
        time_now = int(sys.argv[5])
        time_local = time.localtime(time_now)
        time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

        # it must make sure the sys.arg[2] is exist
        try:
            #TODO
            trigger = LoadPerformance(time_sql, mysql_w, model, batch_size, gpu_card)
        except Exception as e:
            print ("[error]: Pls Check The Modle input wrong!")
            sys.exit(1)
    else:
        print ("[error]: Pls Check args!")
        sys.exit(1)

    trigger.analysis_anakin2_qps()
    trigger.analysis_tensorRT_qps()
