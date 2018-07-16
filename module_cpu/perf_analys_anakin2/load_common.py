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
    def __init__(self, model, thread_size, cpu_card):
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

            self.cpu_card = cpu_card
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

    def create_table_sql_anakin2_qps(self):
        """
        create the table of anakin2_yolo_time_satistic
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS anakin2_yolo_time_satistic_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    latency float not null,
                                    qps float not null
                                    )""" % self.cpu_card
        logging.info("start create the table : anakin2_yolo_time_satistic")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table anakin2_yolo_time_satistic_%s sucess~~" % self.cpu_card)
        except Exception as exception:
            logging.error("create the table anakin2_yolo_time_satistic_%s failed~~" % self.cpu_card)

    def create_table_sql_paddle_qps(self):
        """
        create the table of paddle_time_satistic
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS paddle_time_satistic_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    latency float not null,
                                    qps float not null
                                    )""" % self.cpu_card
        logging.info("start create the table : paddle_time_satistic")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table paddle_time_satistic_%s sucess~~" % self.cpu_card)
        except Exception as exception:
            logging.error("create the table paddle_time_satistic_%s failed~~" % self.cpu_card)

    def create_table_sql_lego_qps(self):
        """
        create the table of lego_time_satistic
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS lego_time_satistic_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    latency float not null,
                                    qps float not null
                                    )""" % self.cpu_card
        logging.info("start create the table : lego_time_satistic")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table lego_time_satistic_%s sucess~~" % self.cpu_card)
        except Exception as exception:
            logging.error("create the table lego_time_satistic_%s failed~~" % self.cpu_card)

    def create_table_sql_top(self):
        """
        create the table of top_list_1sec
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS top_list_1sec_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    cpu_ratio float not null default '0.0',
                                    phy_mem_size float not null default '0.0',
                                    vir_mem_size float not null default '0.0'
                                    )""" % self.cpu_card
        logging.info("start create the table : top_list_1sec")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table top_list_1sec_%s sucess~~" % self.cpu_card)
        except Exception as exception:
            logging.error("create the table top_list_1sec_%s failed~~" % self.cpu_card)

    def create_table_sql_top_avg(self):
        """
        create the table of top_list_1sec_avg
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS top_list_1sec_avg_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL UNIQUE,
                                    cpu_ratio_avg float not null default '0.0',
                                    phy_mem_size_avg float not null default '0.0',
                                    vir_mem_size_avg float not null default '0.0'
                                    )""" % self.cpu_card
        logging.info("start create the table : top_list_1sec_avg")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table top_list_1sec_avg_%s sucess~~" % self.cpu_card)
        except Exception as exception:
            logging.error("create the table top_list_1sec_avg_%s failed~~" % self.cpu_card)


    def insert_table_sql_anakin2_qps(self, time, latency, qps):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO anakin2_yolo_time_satistic_%s(time, latency, qps) VALUES ('%s', '%f', '%f')""" % (self.cpu_card, time, latency, qps)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into anakin2_yolo_time_satistic_%s success!!!" % self.cpu_card)
        except Exception as exception:
            logging.error("insert into anakin2_yolo_time_satistic_%s failed!!!" % self.cpu_card)

    def insert_table_sql_paddle_qps(self, time, latency, qps):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO paddle_time_satistic_%s(time, latency, qps) VALUES ('%s', '%f', '%f')""" % (self.cpu_card, time, latency, qps)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into paddle_time_satistic_%s success!!!" % self.cpu_card)
        except Exception as exception:
            logging.error("insert into paddle_time_satistic_%s failed!!!" % self.cpu_card)

    def insert_table_sql_lego_qps(self, time, latency, qps):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO lego_time_satistic_%s(time, latency, qps) VALUES ('%s', '%f', '%f')""" % (self.cpu_card, time, latency, qps)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into lego_time_satistic_%s success!!!" % self.cpu_card)
        except Exception as exception:
            logging.error("insert into lego_time_satistic_%s failed!!!" % self.cpu_card)

    def insert_table_sql_top(self, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO top_list_1sec_%s(time, cpu_ratio, phy_mem_size, vir_mem_size) VALUES ('%s', '%f', '%f', '%f')""" % (self.cpu_card, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into top_list_1sec_%s success!!!" % self.cpu_card)
        except Exception as exception:
            logging.error("insert into top_list_1sec_%s failed!!!" % self.cpu_card)

    def insert_table_sql_top_avg(self, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO top_list_1sec_avg_%s(time, cpu_ratio_avg, phy_mem_size_avg, vir_mem_size_avg) VALUES ('%s', '%f', '%f', '%f')""" % (self.cpu_card, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into top_list_1sec_avg_%s success!!!" % self.cpu_card)
        except Exception as exception:
            logging.error("insert into top_list_1sec_avg_%s failed!!!" % self.cpu_card)


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
