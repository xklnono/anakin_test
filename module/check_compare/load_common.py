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
    def __init__(self, model, batch_size, gpu_card):
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
                self.test_db=cf.get(conf_name, "test_db") % batch_size
            except Exception as e:
                print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
                sys.exit(1)

            self.mysql = mysql_helper.MysqlHelper(host = self.mysql_host,\
                port = self.mysql_port, user = self.mysql_user, \
                passwd = self.mysql_passwd, db = self.test_db)

            self.gpu_card = gpu_card
        except Exception as exception: 
            print exception
            return     

    def create_table_sql(self):
        """
        create the table of idw_data_delay
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS log_monitor_%s(
                                    version_id int(5) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL UNIQUE,
                                    total_num int(6) not null,
                                    pass_num int(6) not null default '0',
                                    no_pass_num int(6) not null default '0',
                                    success_ratio float(5,4) not null default '1',
                                    no_right_shape int(6) default '0',
                                    no_right_data int(6) default '0',
                                    no_right_weight int(6) default '0',
                                    no_right_c_and_h int(6) default '0'
                                    )""" % self.gpu_card
        logging.info("start create the table : log_monitor")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table log_monitor_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table log_monitor_%s failed~~" % self.gpu_card)

    def insert_table_sql(self, time, total_num, pass_num, no_pass_num, success_ratio, no_right_shape=0, no_right_data=0, no_right_weight=0, no_right_c_and_h=0):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO log_monitor_%s(time, total_num, pass_num, no_pass_num, success_ratio, no_right_shape, no_right_data, no_right_weight, no_right_c_and_h) VALUES ('%s', '%d', '%d', '%d', '%f', '%d', '%d', '%d', '%d')""" % (self.gpu_card, time, total_num, pass_num, no_pass_num, success_ratio, no_right_shape, no_right_data, no_right_weight, no_right_c_and_h)
        print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into log_monitor_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into log_monitor_%s failed!!!" % self.gpu_card)
    
    
    def select_table_sql(self, list_name, value, logging=None):
        """
        execute the select sql
        """
        if list_name != "version_id" and \
            list_name != "time" and \
            list_name != "total_num" and \
            list_name != "pass_num" and \
            list_name != "no_pass_num" and \
            list_name != "no_right_shape" and \
            list_name != "no_right_data" and \
            list_name != "no_right_weight" and \
            list_name != "no_right_c_and_h":

            logging.error("select log_monitor failed!!!")
        else:
            select_sql = """SELECT * from log_monitor WHERE %s = %s""" % (list_name, value)
            print select_sql
        
            logging.info("start instert the sql")
            try:
                select_result = self.mysql.executes(select_sql)
                logging.info("select log_monitor success!!!")
                print select_result
            except Exception as exception:
                logging.error("select log_monitor failed!!!")
