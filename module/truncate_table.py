#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
@Brief entrance of program
"""
import subprocess
import sys
import time
import ConfigParser
import json
import urllib
import urllib2
import MysqlHelper as mysql_helper

def truncate_table_sql(model, batch_size):
    """
    init
    """
    try:
        cf = ConfigParser.ConfigParser()
        cf.read("./conf/load_config.conf")
        mysql_host=cf.get("db", "mysql_host")
        mysql_port=cf.getint("db", "mysql_port")
        mysql_user=cf.get("db", "mysql_user")
        mysql_passwd=cf.get("db", "mysql_passwd")
        conf_name = "conf_%s" % model
        try:
            test_db=cf.get(conf_name, "test_db") % batch_size
        except Exception as e:
            sys.exit(1)
        mysql = mysql_helper.MysqlHelper(host = mysql_host,\
            port = mysql_port, user = mysql_user, \
            passwd = mysql_passwd, db = test_db)

    except Exception as exception: 
        print exception
        return     
 
    table_name=["anakin2_yolo_time_satistic_k1200", "anakin2_yolo_time_satistic_p4", \
                "anakin_tensorrt_time_satistic_k1200", "anakin_tensorrt_time_satistic_p4", \
                "nvidia_list_1sec_k1200", "nvidia_list_1sec_p4", \
                "nvidia_list_1sec_version_k1200", "nvidia_list_1sec_version_p4", \
                "nvidia_list_1sec_version_tensorRT_k1200", "nvidia_list_1sec_version_tensorRT_p4", \
                "top_list_1sec_avg_k1200", "top_list_1sec_avg_p4", \
                "top_list_1sec_avg_tensorRT_k1200", "top_list_1sec_avg_tensorRT_p4", \
                "top_list_1sec_k1200", "top_list_1sec_p4", \
                "log_monitor_k1200", "log_monitor_p4"]
    for item in table_name:
        truncate_sql = "truncate table %s" % (item)
        
        print "[INFO]: start truncate the sql"
        try:
            truncate_result = mysql.executes(truncate_sql)
            print ("[INFO]: truncate %s success!!!" % item)
            print truncate_result
        except Exception as exception:
            print ("[ERROR]: truncate %s error!!!" % item)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        model = sys.argv[1]
        batch_size = sys.argv[2]
        truncate_table_sql(model, batch_size)
    else:
        print "xxxxxxxxxxxxinput error!!!!xxxxxxxxxxxx"
        sys.exit(1)
