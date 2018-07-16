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
    def __init__(self, mysql_w, time_sql, model="", thread_size=1, cpu_card="5117"):
        """
        init
        """
        try:
            cf = ConfigParser.ConfigParser()
            cf.read("../conf/load_config.conf")
            #TODO
            conf_name = "conf_%s" % model
            try:
                #self.filename_top = cf.get(conf_name, "top_result_filename")
                #write dead---no need from config
                self.cpu_card = cpu_card
                self.filename_top = "anakin2_top_result_filename_%s.txt" % self.cpu_card
                self.time_sql = time_sql
            except Exception as e:
                print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
                sys.exit(1)
            
            # get pid in top's file
            pid_set = set()
            temp_f1 = open(self.filename_top)
            for line in temp_f1.readlines():
                if "qa_work" in line:
                    pid_set.add(line.split()[0])
            temp_f1.close()
            if len(pid_set) != 1:
                sys.exit(1)
            self.pid = pid_set.pop()

            self.file_top = open(self.filename_top)

            # top's list
            self.cpu_list_1sec = []
            self.phy_mem_list_1sec = []
            self.virt_mem_list_1sec = []
            self.top_pertime = []

            # init mysql
            self.mysql = LoadCommon(model, thread_size, self.cpu_card)

            # init env: truncate table top_list_1sec 
            self.mysql.create_database()

            self.mysql.create_table_sql_top()
            self.mysql.create_table_sql_top_avg()
            self.db_onoff = mysql_w
            if self.db_onoff == "on":
                self.mysql.truncate_table_sql("top_list_1sec_%s" % self.cpu_card)
        except Exception as exception:
            print exception
            return

    def __del__(self):
        """
        delete
        """
        try:
            self.file_top.close()
        except Exception as exception:
            print exception
            return

    def analysis_host_sec(self):
        """
        analysis the performace which in top cmd 
        1. host(cpu) usage percent 
        2. host(mem) physical memory size
        3. host(mem) virtual memory size
        """
        #calc the date
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        date = time.strftime("%Y-%m-%d",time_local)
        sum_cpu_ratio = 0
        sum_phy_mem_size = 0
        sum_virt_mem_size = 0

        key_re_word = "%s qa_work" % self.pid
        for line in self.file_top.readlines():
            if re.search(key_re_word, line):
                #analysis_cpu_rate()
                sum_cpu_ratio += float(line.split()[8])
                self.cpu_list_1sec.append(float(line.split()[8]))

                #analysis_host_phy_mem_size(), the standerd unit is "g"
                if "m" in line.split()[5]:
                    phy_mem_size = float(line.split()[5].strip("m")) / 1000
                elif "g" in line.split()[5]:
                    phy_mem_size = float(line.split()[5].strip("g"))
                elif "k" in line.split()[5]:
                    phy_mem_size = float(line.split()[5].strip("k")) / 1000 / 1000
                else:
                    phy_mem_size = 0.0
                self.phy_mem_list_1sec.append(float(phy_mem_size))
                sum_phy_mem_size += phy_mem_size

                #analysis_host_virt_mem_size(), the standerd unit is "g"
                if "m" in line.split()[4]:
                    vir_mem_size = float(line.split()[4].strip("m")) / 1000
                elif "g" in line.split()[4]:
                    vir_mem_size = float(line.split()[4].strip("g"))
                elif "k" in line.split()[4]:
                    vir_mem_size = float(line.split()[4].strip("k")) / 1000 / 1000
                else:
                    vir_mem_size = 0
                self.virt_mem_list_1sec.append(float(vir_mem_size))
                sum_virt_mem_size += vir_mem_size

            elif re.search("top -", line):
                final_time = date + " " + line.split()[2]
                self.top_pertime.append(final_time)
        top_num = min(len(self.top_pertime), len(self.cpu_list_1sec), len(self.phy_mem_list_1sec), len(self.virt_mem_list_1sec))
        for i in range(top_num):
            print "%s: %s, %s, %s" % (self.top_pertime[i], self.cpu_list_1sec[i], self.phy_mem_list_1sec[i], self.virt_mem_list_1sec[i])
            #insert into mysql-top_list_1sec
            if self.db_onoff == "on":
                # if the monitor 60min > time > 30min, make the interval from 1s into 2s
                if top_num < 1800:
                    self.mysql.insert_table_sql_top(self.top_pertime[i], self.cpu_list_1sec[i], self.phy_mem_list_1sec[i], self.virt_mem_list_1sec[i])
                elif top_num > 1800 and top_num < 3600:
                    if i % 2 == 0:
                        self.mysql.insert_table_sql_top(self.top_pertime[i], self.cpu_list_1sec[i], self.phy_mem_list_1sec[i], self.virt_mem_list_1sec[i])
                elif top_num > 3600 and top_num < 7200:
                    if i % 4 == 0:
                        self.mysql.insert_table_sql_top(self.top_pertime[i], self.cpu_list_1sec[i], self.phy_mem_list_1sec[i], self.virt_mem_list_1sec[i])
                elif top_num > 7200:
                    if i % 6 == 0:
                        self.mysql.insert_table_sql_top(self.top_pertime[i], self.cpu_list_1sec[i], self.phy_mem_list_1sec[i], self.virt_mem_list_1sec[i])

        #cal the average data
        average_cpu_ratio = round(sum_cpu_ratio/len(self.cpu_list_1sec), 2)
        average_phy_mem_size = round(sum_phy_mem_size/len(self.phy_mem_list_1sec), 2)
        average_virt_mem_size = round(sum_virt_mem_size/len(self.virt_mem_list_1sec), 2)
        #cal the max data
        max_cpu_ratio = max(self.cpu_list_1sec)
        max_phy_mem_size = max(self.phy_mem_list_1sec)
        max_virt_mem_size = max(self.virt_mem_list_1sec)
        #insert into mysql-top_list_1sec_avg
        print "average_cpu_ratio: %s" % average_cpu_ratio
        print "average_phy_mem_size: %s" % average_phy_mem_size
        print "average_virt_mem_size: %s" % average_virt_mem_size
        print "max_cpu_ratio: %s" % max_cpu_ratio
        print "max_phy_mem_size: %s" % max_phy_mem_size
        print "max_virt_mem_size: %s" % max_virt_mem_size
        if self.db_onoff == "on":
            #self.mysql.insert_table_sql_top_avg(self.top_pertime[top_num-1], average_cpu_ratio, average_phy_mem_size, average_virt_mem_size)
            #self.mysql.insert_table_sql_top_avg(self.top_pertime[top_num-1], max_cpu_ratio, max_phy_mem_size, max_virt_mem_size)
            # use time_id
            self.mysql.insert_table_sql_top_avg(self.time_sql, max_cpu_ratio, max_phy_mem_size, max_virt_mem_size)
      

if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    global MYSQL_DB_ONOFF

    time_now = int(time.time())
    time_local = time.localtime(time_now)
    time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    if len(sys.argv) == 1:
        mysql_w = MYSQL_DB_ONOFF
        trigger = LoadPerformance(mysql_w, time_sql, "text_classification", 1, "5117")
    elif len(sys.argv) == 2:
        mysql_w = sys.argv[1]
        trigger = LoadPerformance(mysql_w, time_sql, "text_classification", 1, "5117")
    elif len(sys.argv) == 3:
        #TODO
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        thread_size = 1
        cpu_card = "5117"
        # it must make sure the sys.arg[2] is exist
        try:
            trigger = LoadPerformance(mysql_w, time_sql, model, thread_size, cpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 4:
        #TODO
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        thread_size = sys.argv[3]
        cpu_card = "5117"
        # it must make sure the sys.arg[2] is exist
        try:
            trigger = LoadPerformance(mysql_w, time_sql, model, thread_size, cpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 5:
        #TODO
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        thread_size = sys.argv[3]
        cpu_card = sys.argv[4]
        # it must make sure the sys.arg[2] is exist
        try:
            trigger = LoadPerformance(mysql_w, time_sql, model, thread_size, cpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 6:
        #TODO
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
            trigger = LoadPerformance(mysql_w, time_sql, model, thread_size, cpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)

    trigger.analysis_host_sec()

    del trigger
