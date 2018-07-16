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


    def select_paddle_latency(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT latency from paddle_time_satistic_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select paddle_time_satistic_%s success!!!" % cpu_card) 
            return select_result
        except Exception as exception:
            logging.error("select paddle_time_satistic_%s failed!!!" % cpu_card) 

    def select_paddle_qps(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT qps from paddle_time_satistic_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select paddle_time_satistic_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select paddle_time_satistic_%s failed!!!" % cpu_card)

    def select_paddle_ratio(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT model_paddle_cpu_ratio_avg from top_list_1sec_paddle_avg_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select top_list_1sec_paddle_avg_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select top_list_1sec_paddle_avg_%s failed!!!" % cpu_card)

    def select_paddle_memory(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT model_paddle_phy_mem_size_avg from top_list_1sec_paddle_avg_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select top_list_1sec_paddle_avg_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select top_list_1sec_paddle_avg_%s failed!!!" % cpu_card)

    def select_anakin2_latency(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT latency from anakin2_yolo_time_satistic_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select anakin2_yolo_time_satistic_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select anakin2_yolo_time_satistic_%s failed!!!" % cpu_card)

    def select_anakin2_qps(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT qps from anakin2_yolo_time_satistic_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select anakin2_yolo_time_satistic_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select anakin2_yolo_time_satistic_%s failed!!!" % cpu_card)

    def select_anakin2_ratio(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT cpu_ratio_avg from top_list_1sec_avg_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select top_list_1sec_avg_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select top_list_1sec_avg_%s failed!!!" % cpu_card)

    def select_anakin2_memory(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT phy_mem_size_avg from top_list_1sec_avg_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select top_list_1sec_avg_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select top_list_1sec_avg_%s failed!!!" % cpu_card)

    def select_lego_latency(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT latency from lego_time_satistic_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select lego_time_satistic_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select lego_time_satistic_%s failed!!!" % cpu_card)

    def select_lego_qps(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT qps from lego_time_satistic_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select lego_time_satistic_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select lego_time_satistic_%s failed!!!" % cpu_card)

    def select_lego_ratio(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT model_lego_cpu_ratio_avg from top_list_1sec_lego_avg_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select top_list_1sec_lego_avg_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select top_list_1sec_lego_avg_%s failed!!!" % cpu_card)

    def select_lego_memory(self, cpu_card):
        """
        execute the select sql
        """
        select_sql = """SELECT model_lego_phy_mem_size_avg from top_list_1sec_lego_avg_%s order by time desc limit 1;""" % cpu_card

        try:
            # this is for select result is tulple to string
            select_result = self.mysql.executes(select_sql)[0][0]
            logging.info("select top_list_1sec_lego_avg_%s success!!!" % cpu_card)
            return select_result
        except Exception as exception:
            logging.error("select top_list_1sec_lego_avg_%s failed!!!" % cpu_card)


