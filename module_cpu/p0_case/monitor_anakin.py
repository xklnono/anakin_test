#!/usr/bin/python
# -*- coding=utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Doing trigger: ./net_exec_test_yolo and get cmd(top) info

Authors: sysqa(sysqa@baidu.com)
Date:    2018/04/09
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

GLOBAL_TIME_INTERVAL = 1

def get_monitor_prog_pid(model, cpu_card):
    conf_name = "conf_%s" % model
    try:
        cmd = cf.get(conf_name, "p0_anakin2_ps_cmd") % cpu_card
    except Exception as e:
        print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
        sys.exit(1)
    pid = commands.getoutput(cmd)
    if pid == "":
        return -1
    else:
        return pid

def monitor_anakin_prog(time_interval, top_result_file, pid):
#    # cmd1: top
#    cmd = "nohup top -b -d %d -p %s > %s &" % (time_interval, pid, top_result_file)
#    os.system(cmd)
#    # cmd2: nvidia-smi
#    cmd = "nohup ./nv-smi %s %s &" % (time_interval, cpu_result_file)
#    os.system(cmd)

    while True:
        # check the main anakin2.0 UT test is over or not
        check_cmd = "ps -ef|grep \" %s \"|grep -v grep|grep -v top" % pid
        check_result = commands.getoutput(check_cmd)
        if check_result == "":
            break
        time.sleep(time_interval)
        print "=======ing======="
#    # cmd1: kill top
#    check_cmd_top = "ps -ef|grep %s|grep 'top -b -d'|grep -v grep|awk {'print $2'}" % pid
#    kill_check_pid = commands.getoutput(check_cmd_top)
#    for one_pid in kill_check_pid.split("\n"):
#        kill_check_cmd = "kill -9 %s" % one_pid
#        os.system(kill_check_cmd)
#    # cmd2: kill nvidia-smi
#    check_cmd_nvidiasmi = "ps -ef|grep 'nv-smi %s %s'|grep -v grep|awk {'print $2'}" % (time_interval, gpu_result_file)
#    kill_check_pid = commands.getoutput(check_cmd_nvidiasmi)
#    for one_pid in kill_check_pid.split("\n"):
#        kill_check_cmd = "kill -9 %s" % one_pid
#        os.system(kill_check_cmd)
    print "vvvvvvvvvvvvvvvvvvvvv"
   

def jorcold_start_test_yolo(top_result_file, ut_yolo_path, jorcold_start_cmd, model, cpu_card):
    """
    Start The UT Test In yolo Module
    """
    current_path = os.getcwd()

    ut_anakin2_path = ut_yolo_path
    jorcold_start = jorcold_start_cmd
    # 1.change path to the anakin2.0 ut bin path
    os.chdir(ut_anakin2_path)
    os.system(jorcold_start)

    # 1.change path to pwd
    os.chdir(current_path)
    pid = get_monitor_prog_pid(model, cpu_card)
    if pid == -1:
        sys.exit(1)
        print "[error]: get_monitor_prog_pid fail, return -1"
    elif pid == -2:
        print "[error]: get_monitor_prog_pid fail, return -2"
        sys.exit(1)
    time_interval = GLOBAL_TIME_INTERVAL
    print pid
    monitor_anakin_prog(time_interval, top_result_file, pid)
    
if __name__ == '__main__':
    # init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    cf = ConfigParser.ConfigParser()
    cf.read("../conf/load_config.conf")
    if len(sys.argv) == 4:
        #TODO
        model = sys.argv[1]
        thread_size = sys.argv[2]
        cpu_card = sys.argv[3]
        conf_name = "conf_%s" % model
        try:
            #write dead---no need from config
            #top_result_file = cf.get(conf_name, "top_result_filename")
            top_result_file = "anakin2_top_result_filename_%s.txt" % cpu_card
            #ut_yolo_path = cf.get(conf_name, "anakin2_ut_yolo_path")
            ut_yolo_path = "/home/qa_work/CI/workspace/sys_anakin_merge_build/output"
            jorcold_start_cmd = cf.get(conf_name, "p0_anakin2_jorcold_start_cmd") % (cpu_card, thread_size)
        except Exception as e:
            print ("\033[0;31;m[error]: 111Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    if len(sys.argv) == 3:
        #TODO
        model = sys.argv[1]
        thread_size = sys.argv[2]
        cpu_card = "5117"
        conf_name = "conf_%s" % model
        try:
            #write dead---no need from config
            #top_result_file = cf.get(conf_name, "top_result_filename")
            top_result_file = "anakin2_top_result_filename_%s.txt" % cpu_card
            #ut_yolo_path = cf.get(conf_name, "anakin2_ut_yolo_path")
            ut_yolo_path = "/home/qa_work/CI/workspace/sys_anakin_merge_build/output"
            jorcold_start_cmd = cf.get(conf_name, "p0_anakin2_jorcold_start_cmd") % (cpu_card, thread_size)
        except Exception as e:
            print ("\033[0;31;m[error]: 111Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 2:
        #TODO
        # if not input thread_size, we use thread_size=1
        thread_size = 1
        model = sys.argv[1]
        cpu_card = "5117"
        conf_name = "conf_%s" % model
        try:
            #top_result_file = cf.get(conf_name, "top_result_filename")
            #write dead---no need from config
            top_result_file = "anakin2_top_result_filename_%s.txt" % cpu_card
            #ut_yolo_path = cf.get(conf_name, "anakin2_ut_yolo_path")
            ut_yolo_path = "/home/qa_work/CI/workspace/sys_anakin_merge_build/output"
            jorcold_start_cmd = cf.get(conf_name, "p0_anakin2_jorcold_start_cmd") % (cpu_card, thread_size)
        except Exception as e:
            print ("\033[0;31;m[error]: 222Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 1:
        #top_result_file = cf.get("conf_yolo", "top_result_filename")
        #write dead---no need from config
        # if not input thread_size, we use thread_size=1
        thread_size = 1
        model = "text_classification"
        cpu_card = "5117"
        top_result_file = "anakin2_top_result_filename_%s.txt" % cpu_card
        #ut_yolo_path = cf.get("conf_yolo", "anakin2_ut_yolo_path")
        ut_yolo_path = "/home/qa_work/CI/workspace/sys_anakin_merge_build/output"
        jorcold_start_cmd = cf.get("conf_yolo", "p0_anakin2_jorcold_start_cmd") % (cpu_card, thread_size)
    else:
        print ("[error]: input error!")

    jorcold_start_test_yolo(top_result_file, ut_yolo_path, jorcold_start_cmd, model, cpu_card)

