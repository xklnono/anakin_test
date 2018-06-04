#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
@Brief entrance of program
"""
import subprocess
import time
import ConfigParser
import json
import urllib
import urllib2
import MysqlHelper as mysql_helper
import logging

import mylogging

class LoadCommon(object):
    """
    load config
    """
    def __init__(self, db_name):
        """
        init
        """
        self.config = {}
        try:
            cf = ConfigParser.ConfigParser()
            cf.read("../conf/load_config.conf")
            self.mysql_host=cf.get("db", "mysql_host")
            self.mysql_port=cf.getint("db", "mysql_port")
            self.mysql_user=cf.get("db", "mysql_user")
            self.mysql_passwd=cf.get("db", "mysql_passwd")
            self.test_db = db_name
            self.mysql = mysql_helper.MysqlHelper(host = self.mysql_host,\
                port = self.mysql_port, user = self.mysql_user, \
                passwd = self.mysql_passwd, db = self.test_db)

        except Exception as exception: 
            print exception
            return     

    def truncate_table_sql(self, table_name):
        """
        execute the truncate sql
        """
        truncate_sql = "truncate table %s" % (table_name)
    
        logging.info("start truncate the sql")
        try:
            truncate_result = self.mysql.executes(truncate_sql)
            logging.info("truncate %s success!!!" % table_name)
            print truncate_result
        except Exception as exception:
            logging.error("truncate %s error!!!" % table_name)


    def select_tensorRT_latency(self, gpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT tensorrt_average_time from anakin_tensorrt_time_satistic_%s order by time desc limit 1;""" % gpu_card

        try:
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select anakin_tensorrt_time_satistic_%s success!!!" % gpu_card) 
            return select_result
        except Exception as exception:
            logging.error("select anakin_tensorrt_time_satistic_%s failed!!!" % gpu_card) 

    def select_tensorRT_memory(self, gpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT model_tensorRT_mem_size_avg from nvidia_list_1sec_version_tensorRT_%s order by time desc limit 1;""" % gpu_card

        try:
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select nvidia_list_1sec_version_tensorRT_%s success!!!" % gpu_card)
            return select_result
        except Exception as exception:
            logging.error("select nvidia_list_1sec_version_tensorRT_%s failed!!!" % gpu_card)

    def select_anakin2_latency(self, gpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT average_time from anakin2_yolo_time_satistic_%s order by time desc limit 1;""" % gpu_card

        try:
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select anakin_tensorrt_time_satistic_%s success!!!" % gpu_card)
            return select_result
        except Exception as exception:
            logging.error("select anakin_tensorrt_time_satistic_%s failed!!!" % gpu_card)

    def select_anakin2_memory(self, gpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT mem_size_avg from nvidia_list_1sec_version_%s order by time desc limit 1;""" % gpu_card

        try:
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select nvidia_list_1sec_version_%s success!!!" % gpu_card)
            return select_result
        except Exception as exception:
            logging.error("select nvidia_list_1sec_version_%s failed!!!" % gpu_card)

