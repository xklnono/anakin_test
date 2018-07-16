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
    def __init__(self, mysql_w, time_sql, model="", batch_size=1, gpu_card="p4"):
        """
        init
        """
        try:
            cf = ConfigParser.ConfigParser()
            #cf.read("./conf/load_config.conf")
            cf.read("../conf/load_config.conf")
            #TODO
            conf_name = "conf_%s" % model
            try:
                #self.filename_top = cf.get(conf_name, "top_result_filename")
                #self.filename_gpu = cf.get(conf_name, "gpu_result_filename")
                #write dead---no need from config
                self.gpu_card = gpu_card
                self.filename_top = "anakin2_top_result_filename_%s.txt" % self.gpu_card
                self.filename_gpu = "anakin2_gpu_result_filename_%s.txt" % self.gpu_card
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
            self.file_gpu = open(self.filename_gpu)

            # top's list
            self.cpu_list_1sec = []
            self.phy_mem_list_1sec = []
            self.virt_mem_list_1sec = []
            self.top_pertime = []

            # gpu's list
            self.gpu_pertime = []

            self.gpu_usage_percent_1 = []
            self.gpu_usage_percent_2 = []
            self.gpu_usage_percent_3 = []
            self.gpu_usage_percent_4 = []
            self.gpu_usage_percent_all = []

            self.gpu_mem_1 = []
            self.gpu_mem_2 = []
            self.gpu_mem_3 = []
            self.gpu_mem_4 = []
            self.gpu_mem_all = []

            self.gpu_temper_1 = []
            self.gpu_temper_2 = []
            self.gpu_temper_3 = []
            self.gpu_temper_4 = []
            self.gpu_temper_max = []

            # init mysql
            self.mysql = LoadCommon(model, batch_size, self.gpu_card)

            # init env: truncate table top_list_1sec 
            self.mysql.create_database()

            self.mysql.create_table_sql_top()
            self.mysql.create_table_sql_nvidia()
            self.mysql.create_table_sql_top_avg()
            self.mysql.create_table_sql_nvidia_version()
            self.db_onoff = mysql_w
            if self.db_onoff == "on":
                self.mysql.truncate_table_sql("top_list_1sec_%s" % self.gpu_card)
                self.mysql.truncate_table_sql("nvidia_list_1sec_%s" % self.gpu_card)
        except Exception as exception:
            print exception
            return

    def __del__(self):
        """
        delete
        """
        try:
            self.file_top.close()
            self.file_gpu.close()
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
      
    def analysis_dev_sec(self):
        """
        analysis the device(gpu) physical memory size
        analysis the performace which in nvidia-smi cmd
        1. device(gpu) physical memory size
        2. device(gpu) usage percent
        3. device(gpu) temperature
        """
        #calc the date
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        date = time.strftime("%Y-%m-%d",time_local)
        sum_cpu_ratio = 0
        sum_gpu_mem_size = 0
        # key: time key
        key_re_time = "[0-9]+ [0-9]+:[0-9]+:[0-9]+ 20[12][][0-9]"
        # key: temperature key
        key_re_temper = "[0-9]+C"
        # key: gpu percent key
        key_re_percent = "[0-9]+%"
        # key: gpu mem key
        key_re_mem = "%s" % self.pid
        key_re_mem_null = "No running processes found"
        # key: line ending key
        key_ending = "====ending===="

        new_gpu_data_count = 0
        sum_gpu_usage_percent_all = 0
        for line in self.file_gpu.readlines():
            if re.search(key_re_time, line):
                # time own unit
                # 1. colect the gpu time info
                final_time = date + " " + line.split()[3]
                self.gpu_pertime.append(final_time)
            elif re.search(key_re_temper, line) and re.search(key_re_percent, line):
                #print "2222, data_line: %s" % line
                # 2. colect the gpu temperature info
                # 3. colect the gpu usage percentage info
                temper = float(line.split()[2].rstrip("C"))
                gpu_usage = float(line.split()[12].rstrip("%"))
                if new_gpu_data_count == 0:
                    self.gpu_temper_1.append(temper)
                    self.gpu_usage_percent_1.append(gpu_usage)
                elif new_gpu_data_count == 1:
                    self.gpu_temper_2.append(temper)
                    self.gpu_usage_percent_2.append(gpu_usage)
                elif new_gpu_data_count == 2:
                    self.gpu_temper_3.append(temper)
                    self.gpu_usage_percent_3.append(gpu_usage)
                elif new_gpu_data_count == 3:
                    self.gpu_temper_4.append(temper)
                    self.gpu_usage_percent_4.append(gpu_usage)
                new_gpu_data_count += 1
            elif re.search(key_re_mem, line) or re.search(key_re_mem_null, line):
                # 4. colect the gpu mem info
                this_gpu_num = line.split()[1]
                if "MiB" in line.split()[5]:
                    this_gpu_mem = float(line.split()[5].strip("MiB"))
                # TODO_this: if there have other unit

                if this_gpu_num == "0":
                    self.gpu_mem_1.append(this_gpu_mem)
                elif this_gpu_num == "1":
                    self.gpu_mem_2.append(this_gpu_mem)
                elif this_gpu_num == "2":
                    self.gpu_mem_3.append(this_gpu_mem)
                elif this_gpu_num == "3":
                    self.gpu_mem_4.append(this_gpu_mem)
                elif this_gpu_num == "No":
                    self.gpu_mem_1.append(0)
                    self.gpu_mem_2.append(0)
                    self.gpu_mem_3.append(0)
                    self.gpu_mem_4.append(0)
                
            elif re.search(key_ending, line):
                # control unit
                # 1.complete the gpu_mem list
                max_len_gpu_mem = max(len(self.gpu_mem_4), len(self.gpu_mem_3), len(self.gpu_mem_2), len(self.gpu_mem_1))
                min_len_gpu_mem = min(len(self.gpu_mem_4), len(self.gpu_mem_3), len(self.gpu_mem_2), len(self.gpu_mem_1))
                if max_len_gpu_mem != min_len_gpu_mem:
                    if len(self.gpu_mem_1) != max_len_gpu_mem:
                        self.gpu_mem_1.append(0)
                    if len(self.gpu_mem_2) != max_len_gpu_mem:
                        self.gpu_mem_2.append(0)
                    if len(self.gpu_mem_3) != max_len_gpu_mem:
                        self.gpu_mem_3.append(0)
                    if len(self.gpu_mem_4) != max_len_gpu_mem:
                        self.gpu_mem_4.append(0)
                new_gpu_data_count = 0

        # ! because all the list is equal
        for i in range(len(self.gpu_mem_1)):
            self.gpu_usage_percent_all.append(self.gpu_usage_percent_1[i] + self.gpu_usage_percent_2[i] + self.gpu_usage_percent_3[i] + self.gpu_usage_percent_4[i])

            #self.gpu_mem_all.append(self.gpu_mem_1[i] + self.gpu_mem_2[i] + self.gpu_mem_3[i] + self.gpu_mem_4[i])
            self.gpu_mem_all.append(max(self.gpu_mem_1[i], self.gpu_mem_2[i], self.gpu_mem_3[i], self.gpu_mem_4[i]))
            sum_gpu_mem_size += max(self.gpu_mem_1[i], self.gpu_mem_2[i], self.gpu_mem_3[i], self.gpu_mem_4[i])

            self.gpu_temper_max.append(max(self.gpu_temper_1[i] ,self.gpu_temper_2[i] ,self.gpu_temper_3[i] ,self.gpu_temper_4[i]))

        version_gpu_usage_percent_all = max(self.gpu_usage_percent_all)

        version_gpu_mem_max = max(self.gpu_mem_all)
        version_gpu_mem_avg = round(sum_gpu_mem_size/len(self.gpu_mem_all), 2)

        version_gpu_temper_max = max(self.gpu_temper_max)

        print "version_gpu_usage_percent_all: %s" % version_gpu_usage_percent_all
        print "version_gpu_mem_max: %s" % version_gpu_mem_max
        print "version_gpu_mem_avg: %s" % version_gpu_mem_avg
        print "version_gpu_temper_max: %s" % version_gpu_temper_max

        # insert into database: nvidia_list_1sec
        if self.db_onoff == "on":
            for i in range(len(self.gpu_mem_1)):
                # insert into database: nvidia_list_1sec
                self.mysql.insert_table_sql_nvidia(self.gpu_pertime[i], \
                        self.gpu_usage_percent_1[i], self.gpu_usage_percent_2[i], self.gpu_usage_percent_3[i], self.gpu_usage_percent_4[i],\
                        self.gpu_mem_1[i], self.gpu_mem_2[i], self.gpu_mem_3[i], self.gpu_mem_4[i],\
                        self.gpu_temper_1[i], self.gpu_temper_2[i], self.gpu_temper_3[i], self.gpu_temper_4[i])
            # insert into database: nvidia_list_1sec_avg
            self.mysql.insert_table_sql_nvidia_version(self.time_sql, version_gpu_usage_percent_all, version_gpu_mem_avg, version_gpu_temper_max)
            # insert into database: nvidia_list_1sec_max
            #self.mysql.insert_table_sql_nvidia_version(self.time_sql, version_gpu_usage_percent_all, version_gpu_mem_max, version_gpu_temper_max)

       
            

#        for i in range(len(self.gpu_mem_1)):
#            print "----------------"
#            print self.gpu_temper_1[i], self.gpu_temper_2[i], self.gpu_temper_3[i], self.gpu_temper_4[i]
#            print self.gpu_usage_percent_all[i], self.gpu_usage_percent_1[i], self.gpu_usage_percent_2[i], self.gpu_usage_percent_3[i], self.gpu_usage_percent_4[i]
#            print self.gpu_mem_all[i], self.gpu_mem_1[i], self.gpu_mem_2[i], self.gpu_mem_3[i], self.gpu_mem_4[i]


if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    global MYSQL_DB_ONOFF

    time_now = int(time.time())
    time_local = time.localtime(time_now)
    time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    if len(sys.argv) == 2:
        mysql_w = sys.argv[1]
        trigger = LoadPerformance(mysql_w, time_sql, "yolo", 1, "p4")
    elif len(sys.argv) == 1:
        mysql_w = MYSQL_DB_ONOFF
        trigger = LoadPerformance(mysql_w, time_sql, "yolo", 1, "p4")
    elif len(sys.argv) == 3:
        #TODO
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        batch_size = 1
        gpu_card = "p4"
        # it must make sure the sys.arg[2] is exist
        try:
            trigger = LoadPerformance(mysql_w, time_sql, model, batch_size, gpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 4:
        #TODO
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        batch_size = sys.argv[3]
        gpu_card = "p4"
        # it must make sure the sys.arg[2] is exist
        try:
            trigger = LoadPerformance(mysql_w, time_sql, model, batch_size, gpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 5:
        #TODO
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        batch_size = sys.argv[3]
        gpu_card = sys.argv[4]
        # it must make sure the sys.arg[2] is exist
        try:
            trigger = LoadPerformance(mysql_w, time_sql, model, batch_size, gpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 6:
        #TODO
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
            trigger = LoadPerformance(mysql_w, time_sql, model, batch_size, gpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)

    trigger.analysis_host_sec()
    trigger.analysis_dev_sec()

    del trigger
