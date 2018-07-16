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

    def create_table_sql_anakin2_yolo_qps(self):
        """
        create the table of anakin2_yolo_time_satistic
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS anakin2_yolo_time_satistic_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    image_num int(6) not null,
                                    total_time float not null,
                                    average_time float not null
                                    )""" % self.gpu_card
        logging.info("start create the table : anakin2_yolo_time_satistic")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table anakin2_yolo_time_satistic_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table anakin2_yolo_time_satistic_%s failed~~" % self.gpu_card)

    def create_table_sql_tensorrt_qps(self):
        """
        create the table of anakin_tensorrt_time_satistic
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS anakin_tensorrt_time_satistic_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    tensorrt_image_num int(6) not null,
                                    tensorrt_total_time float not null,
                                    tensorrt_average_time float not null
                                    )""" % self.gpu_card
        logging.info("start create the table : anakin_tensorrt_time_satistic")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table anakin_tensorrt_time_satistic_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table anakin_tensorrt_time_satistic_%s failed~~" % self.gpu_card)

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
                                    )""" % self.gpu_card
        logging.info("start create the table : top_list_1sec")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table top_list_1sec_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table top_list_1sec_%s failed~~" % self.gpu_card)

    def create_table_sql_nvidia(self):
        """
        create the table of nvidia_list_1sec
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS nvidia_list_1sec_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL,
                                    gpu_ratio_1 float not null default '0.0',
                                    gpu_ratio_2 float not null default '0.0',
                                    gpu_ratio_3 float not null default '0.0',
                                    gpu_ratio_4 float not null default '0.0',
                                    mem_size_1 float not null default '0.0',
                                    mem_size_2 float not null default '0.0',
                                    mem_size_3 float not null default '0.0',
                                    mem_size_4 float not null default '0.0',
                                    temperature_1 float not null default '0.0',
                                    temperature_2 float not null default '0.0',
                                    temperature_3 float not null default '0.0',
                                    temperature_4 float not null default '0.0'
                                    )""" % self.gpu_card
        logging.info("start create the table : nvidia_list_1sec")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table nvidia_list_1sec_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table nvidia_list_1sec_%s failed~~" % self.gpu_card)

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
                                    )""" % self.gpu_card
        logging.info("start create the table : top_list_1sec_avg")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table top_list_1sec_avg_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table top_list_1sec_avg_%s failed~~" % self.gpu_card)

    def create_table_sql_nvidia_version(self):
        """
        create the table of nvidia_list_1sec_version
        """
        create_table_sql = """CREATE TABLE IF NOT EXISTS nvidia_list_1sec_version_%s(
                                    num_id int(6) not null primary key AUTO_INCREMENT,
                                    time TIMESTAMP NOT NULL UNIQUE,
                                    ratio_avg float not null default '0.0',
                                    mem_size_avg float not null default '0.0',
                                    temperature_max float not null default '0.0'
                                    )""" % self.gpu_card
        logging.info("start create the table : nvidia_list_1sec_version")
        try:
            self.mysql.execute(create_table_sql)
            logging.info("create the table nvidia_list_1sec_version_%s sucess~~" % self.gpu_card)
        except Exception as exception:
            logging.error("create the table nvidia_list_1sec_version_%s failed~~" % self.gpu_card)

    def insert_table_sql_anakin2_yolo_qps(self, time, image_num, total_time, average_time):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO anakin2_yolo_time_satistic_%s(time, image_num, total_time, average_time) VALUES ('%s', '%d', '%f', '%f')""" % (self.gpu_card, time, image_num, total_time, average_time)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into anakin2_yolo_time_satistic_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into anakin2_yolo_time_satistic_%s failed!!!" % self.gpu_card)

    def insert_table_sql_tensorrt_qps(self, time, image_num, total_time, average_time):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO anakin_tensorrt_time_satistic_%s(time, tensorrt_image_num, tensorrt_total_time, tensorrt_average_time) VALUES ('%s', '%d', '%f', '%f')""" % (self.gpu_card, time, image_num, total_time, average_time)
        print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into anakin_tensorrt_time_satistic_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into anakin_tensorrt_time_satistic_%s failed!!!" % self.gpu_card)

    def insert_table_sql_top(self, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO top_list_1sec_%s(time, cpu_ratio, phy_mem_size, vir_mem_size) VALUES ('%s', '%f', '%f', '%f')""" % (self.gpu_card, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into top_list_1sec_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into top_list_1sec_%s failed!!!" % self.gpu_card)

    def insert_table_sql_nvidia(self, time, gpu_ratio_1, gpu_ratio_2, gpu_ratio_3, gpu_ratio_4, \
                                            mem_size_1, mem_size_2, mem_size_3, mem_size_4, \
                                            temperature_1, temperature_2, temperature_3, temperature_4):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO nvidia_list_1sec_%s(time, gpu_ratio_1, mem_size_1, temperature_1, gpu_ratio_2, mem_size_2, temperature_2, gpu_ratio_3, mem_size_3, temperature_3, gpu_ratio_4, mem_size_4, temperature_4) VALUES ('%s', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f')""" % (self.gpu_card, time, gpu_ratio_1, mem_size_1, temperature_1, gpu_ratio_2, mem_size_2, temperature_2, gpu_ratio_3, mem_size_3, temperature_3, gpu_ratio_4, mem_size_4, temperature_4)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into nvidia_list_1sec_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into nvidia_list_1sec_%s failed!!!" % self.gpu_card)
    
    def insert_table_sql_top_avg(self, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO top_list_1sec_avg_%s(time, cpu_ratio_avg, phy_mem_size_avg, vir_mem_size_avg) VALUES ('%s', '%f', '%f', '%f')""" % (self.gpu_card, top_pertime, cpu_list_1sec, phy_mem_list_1sec, virt_mem_list_1sec)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into top_list_1sec_avg_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into top_list_1sec_avg_%s failed!!!" % self.gpu_card)

    def insert_table_sql_nvidia_version(self, time, ratio_avg, mem_size_avg, temperature_max):
        """
        execute the insert sql
        """
        insert_sql = """INSERT INTO nvidia_list_1sec_version_%s(time, ratio_avg, mem_size_avg, temperature_max) VALUES ('%s', '%f', '%f', '%f')""" % (self.gpu_card, time, ratio_avg, mem_size_avg, temperature_max)
        #print insert_sql
        logging.info("start instert the sql")
        try:
            self.mysql.executes(insert_sql)
            logging.info("insert into nvidia_list_1sec_version_%s success!!!" % self.gpu_card)
        except Exception as exception:
            logging.error("insert into nvidia_list_1sec_version_%s failed!!!" % self.gpu_card)
    
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
