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
    def __init__(self, mysql_w, model="", batch_size=1, gpu_card="p4"):
        """
        init
        """
        try:
            #TODO
            self.multithread_pk_file = "/home/qa_work/CI/workspace/sys_anakin_compare_output/%s/multithread_time/Multi_thread_time.txt" % model
            self.gpu_card = gpu_card
            
            if os.path.exists(self.multithread_pk_file):
                self.file = open(self.multithread_pk_file)
            else:
                print "[error]: the %s does not exist" % self.multithread_pk_file
                sys.exit(1)

            # top's list
            self.batch_size = []
            self.thread_num = []
            self.qps = []

            # init mysql
            self.mysql = LoadCommon(model, batch_size, self.gpu_card)

            # init the initing time
            # the qps table has the same primary key: time
            time_now = int(time.time())
            time_local = time.localtime(time_now)
            self.time_sql = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

            self.mysql.create_table_sql_multithread_qps()
            self.db_onoff = mysql_w
            if self.db_onoff == "on":
                self.mysql.truncate_table_sql("multithread_qps")
        except Exception as exception:
            print exception
            return

    def __del__(self):
        """
        delete
        """
        try:
            self.file.close()
        except Exception as exception:
            print exception
            return

    def analysis_multithread_qps(self):
        """
        analysis the multithread qps
        """
        for line in self.file.readlines():
            batch_size = int(line.split(" ")[3])
            thread_num = int(line.split(" ")[7])
            qps = float(line.split(" ")[11])
            print "batch_size: %s, thread_num: %s, qps: %s" % (batch_size, thread_num, qps)
            self.batch_size.append(batch_size)
            self.thread_num.append(thread_num)
            self.qps.append(qps)
        print self.batch_size
        print self.thread_num
        print self.qps

        if self.db_onoff == "on":
            for i in range(len(self.batch_size)): 
                self.mysql.insert_table_sql_multithread_qps(self.time_sql, self.batch_size[i], self.thread_num[i], self.qps[i])
      

if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    global MYSQL_DB_ONOFF
    if len(sys.argv) == 5:
        #TODO
        mysql_w = sys.argv[1]
        model = sys.argv[2]
        batch_size = sys.argv[3]
        gpu_card = sys.argv[4]
        # it must make sure the sys.arg[2] is exist
        try:
            trigger = LoadPerformance(mysql_w, model, batch_size, gpu_card)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)

    trigger.analysis_multithread_qps()

    del trigger
