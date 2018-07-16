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
    def __init__(self, time_sql, mysql_w, model="", thread_size=1, cpu_card="5117"):
        """
        init
        """
        try:
            cf = ConfigParser.ConfigParser()
            #cf.read("./conf/load_config.conf")
            cf.read("../conf/load_config.conf")
            conf_name = "conf_%s" % model
            try:
                self.cpu_card = cpu_card
                self.time_pk_path = cf.get(conf_name, "time_pk_path")
            except Exception as e:
                print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
                sys.exit(1)
            
            # init mysql
            self.mysql = LoadCommon(model, thread_size, self.cpu_card)

            # init the initing time
            # the qps table has the same primary key: time
            self.time_sql = time_sql

            # init env: truncate table top_list_1sec 
            self.mysql.create_database()
            self.mysql.create_table_sql_anakin2_qps()
            self.mysql.create_table_sql_paddle_qps()
            self.mysql.create_table_sql_lego_qps()
            self.db_onoff = mysql_w
        except Exception as exception:
            print exception
            return

    def analysis_paddle_qps(self):
        """
        analysis the paddle's qps
        """
        paddle_time_file = self.time_pk_path + "/" + "Paddle_time.txt"
        if not os.path.exists(paddle_time_file):
            print "[alarm]: the paddle_time.txt file do not exist"
            sys.exit(0)
        # eg: /home/qa_work/CI/workspace/sys_anakin_compare_output/language/time/Paddle_time.txt
        try:
            f = open(paddle_time_file)
            latency_list = []
            num = 0
            for line in f.readlines():
                if "Tid" in line.split(":")[0]:
                    temp_latency = float(line.split(" ")[-2].strip())
                    latency_list.append(temp_latency)
                    num += 1
                elif "Total" in line.split(":")[0]:
                    qps = float(line.split(" ")[-1].strip())
                else:
                    continue
            sum = 0
            for i in latency_list:
                sum += i
            if num != 0:
                latency = float(sum / num)
            else:
                latency = 0.0
            if self.db_onoff == "on":
                self.mysql.insert_table_sql_paddle_qps(self.time_sql, latency, qps)
                print "paddle-latency: %s, paddle-qps: %s" % (latency, qps)
        except Exception,e:
            print "\033[0;31;m[error]: analysis paddle_time.txt file error\033[0m"
            sys.exit(1)
        finally:
            f.close()
        
    def analysis_lego_qps(self):
        """
        analysis the lego's qps
        """
        lego_time_file = self.time_pk_path + "/" + "Lego_time.txt"
        if not os.path.exists(lego_time_file):
            print "[alarm]: the lego_time.txt file do not exist"
            sys.exit(0)
        try:
            f = open(lego_time_file)
            num = 0
            latency_list = []
            for line in f.readlines():
                if "thread_num_i" in line.split(":")[0]:
                    temp_latency = float(line.split(" ")[-1].strip())
                    latency_list.append(temp_latency)
                    num += 1
                elif "total_time" in line.split(":")[0]:
                    qps = float(line.split(" ")[-1].strip())
                else:
                    continue
            sum = 0
            for i in latency_list:
                sum += i
            if num != 0:
                latency = float(sum / num)
            else:
                latency = 0.0

            if self.db_onoff == "on":
                self.mysql.insert_table_sql_lego_qps(self.time_sql, latency, qps)
        except Exception,e:
            print "\033[0;31;m[error]: analysis lego_time.txt file error\033[0m"
            sys.exit(1)
        finally:
            f.close()

    def analysis_anakin2_qps(self):
        """
        analysis the anakin2.0's qps
        """
        anakin2_time_file = self.time_pk_path + "/" + "Anakin2_time.txt"
        if not os.path.exists(anakin2_time_file):
            print "[alarm]: the Anakin2_time.txt file do not exist"
            sys.exit(0)
        # eg: /home/qa_work/CI/workspace/sys_anakin_compare_output/language/time/Anakin2_time.txt
        try:
            f = open(anakin2_time_file)
            latency_list = []
            num = 0
            for line in f.readlines():
                if "summary_thread" in line.split(":")[0]:
                    temp_latency = float(line.split(",")[1].strip().split("=")[1].strip())
                    latency_list.append(temp_latency)
                    num += 1
                elif "summary" in line.split(":")[0]:
                    qps = float(line.split(" ")[-1].strip())
                else:
                    continue
            sum = 0
            for i in latency_list:
                sum += i
            if num != 0:
                latency = float(sum / num)
            else:
                latency = 0.0
            if self.db_onoff == "on":
                self.mysql.insert_table_sql_anakin2_qps(self.time_sql, latency, qps)
                print "anakin2-latency: %s, anakin2-qps: %s" % (latency, qps)
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
        thread_size = sys.argv[3]
        cpu_card = sys.argv[4]

        # use the same time_now for mysql data
        #time_now = int(time.time())
        time_now = int(sys.argv[5])
        time_local = time.localtime(time_now)
        time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

        # it must make sure the sys.arg[2] is exist
        try:
            #TODO
            trigger = LoadPerformance(time_sql, mysql_w, model, thread_size, cpu_card)
        except Exception as e:
            print ("[error]: Pls Check The Modle input wrong!")
            sys.exit(1)
    else:
        print ("[error]: Pls Check args!")
        sys.exit(1)

    trigger.analysis_anakin2_qps()
    trigger.analysis_paddle_qps()
    trigger.analysis_lego_qps()
