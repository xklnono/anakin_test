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
    def __init__(self, model, thread_size, gpu_card):
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
            #TODO
            conf_name = "conf_%s" % model
            try:
                self.test_db=cf.get(conf_name, "test_db") % thread_size
            except Exception as e:
                self.test_db=cf.get("db", "test_db")
            self.mysql = mysql_helper.MysqlHelper(host = self.mysql_host,\
                port = self.mysql_port, user = self.mysql_user, \
                passwd = self.mysql_passwd, db = self.test_db)

            self.gpu_card = gpu_card
        except Exception as exception: 
            print exception
            return     
 
    def create_database(self):
        """
        create database
        """
        mysql = mysql_helper.MysqlHelper(host = self.mysql_host,\
            port = self.mysql_port, user = self.mysql_user, \
            passwd = self.mysql_passwd)
        create_database_sql = "CREATE DATABASE IF NOT EXISTS %s" % self.test_db
        try:
            mysql.execute_withnodb(create_database_sql)
            logging.info("create the database %s sucess~~" % self.test_db)
        except Exception as exception:
            logging.error("create the database %s failed~~" % self.test_db)

    def create_table_sql_top_avg_model_lego(self):
        """
        create the table of top_list_1sec_avg_lego
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS top_list_1sec_lego_avg_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL UNIQUE,
                                    model_lego_cpu_ratio_avg float not null default '0.0',
                                    model_lego_phy_mem_size_avg float not null default '0.0',
                                    model_lego_vir_mem_size_avg float not null default '0.0'
                                    )""" % self.gpu_card
        logging.info("start create the table : top_list_1sec_avg_lego")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table top_list_1sec_lego_avg_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table top_list_1sec_lego_avg_% failed~~" % self.gpu_card)

    def insert_table_sql_top_avg(self, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO top_list_1sec_lego_avg_%s(time, model_lego_cpu_ratio_avg, model_lego_phy_mem_size_avg, model_lego_vir_mem_size_avg) VALUES ('%s', '%f', '%f', '%f')""" % (self.gpu_card, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into top_list_1sec_lego_avg_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into top_list_1sec_lego_avg_%s failed!!!" % self.gpu_card)

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
