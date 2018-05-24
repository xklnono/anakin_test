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

def get_monitor_prog_pid(model, gpu_card):
    #TODO
    conf_name = "conf_%s" % model
    try:
        cmd = cf.get(conf_name, "tensorrt_ps_cmd") % gpu_card
    except Exception as e:
        print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
        sys.exit(1)
    pid = commands.getoutput(cmd)
    if pid == "":
        return -1
    else:
        return pid

def monitor_anakin_prog(time_interval, gpu_result_file, top_result_file, pid):
#    # cmd1: top
#    cmd = "nohup top -b -d %d -p %s > %s &" % (time_interval, pid, top_result_file)
#    os.system(cmd)
#    # cmd2: nvidia-smi
#    cmd = "nohup ./nv-smi %s %s &" % (time_interval, gpu_result_file)
#    os.system(cmd)

    while True:
        # check the main anakin2.0 UT test is over or not
        check_cmd = "ps -ef|grep \" %s \"|grep -v grep|grep -v top" % pid
        check_result = commands.getoutput(check_cmd)
        if check_result == "":
            break
        #time.sleep(1)
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
   

def jorcold_start_test_yolo(gpu_result_file, top_result_file, ut_yolo_path, jorcold_start_cmd, model, gpu_card):
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
    pid = get_monitor_prog_pid(model, gpu_card)
    if pid == -1:
        sys.exit(1)
    elif pid == -2:
        sys.exit(0)
    print pid
    time_interval = GLOBAL_TIME_INTERVAL
    monitor_anakin_prog(time_interval, gpu_result_file, top_result_file, pid)
    
#    # cp result from output to pwd path
#    # 1.top result
#    cp_cmd = "cp %s %s" % (top_result_file, current_path)
#    os.system(cp_cmd)
#    # 2.nvidia-smi result
#    cp_cmd = "cp %s %s" % (gpu_result_file, current_path)
#    os.system(cp_cmd)

if __name__ == '__main__':
    # init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    cf = ConfigParser.ConfigParser()
    cf.read("../conf/load_config.conf")
    if len(sys.argv) == 4:
        #TODO
        model = sys.argv[1]
        batch_size = sys.argv[2]
        gpu_card = sys.argv[3]
        conf_name = "conf_%s" % model
        try:
            #write dead---no need from config
            #gpu_result_file = cf.get(conf_name, "gpu_result_filename")
            #top_result_file = cf.get(conf_name, "top_result_filename")
            gpu_result_file = "tensorrt_gpu_result_filename_%s.txt" % gpu_card
            top_result_file = "tensorrt_top_result_filename_%s.txt" % gpu_card
            #ut_yolo_path = cf.get(conf_name, "tensorrt_ut_yolo_path")
            ut_yolo_path = "/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output"
            jorcold_start_cmd = cf.get(conf_name, "tensorrt_jorcold_start_cmd") % (gpu_card, batch_size)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    if len(sys.argv) == 3:
        #TODO
        model = sys.argv[1]
        batch_size = sys.argv[2]
        gpu_card = "p4"
        conf_name = "conf_%s" % model
        try:
            #write dead---no need from config
            #gpu_result_file = cf.get(conf_name, "gpu_result_filename")
            #top_result_file = cf.get(conf_name, "top_result_filename")
            gpu_result_file = "tensorrt_gpu_result_filename_%s.txt" % gpu_card
            top_result_file = "tensorrt_top_result_filename_%s.txt" % gpu_card
            #ut_yolo_path = cf.get(conf_name, "tensorrt_ut_yolo_path")
            ut_yolo_path = "/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output"
            jorcold_start_cmd = cf.get(conf_name, "tensorrt_jorcold_start_cmd") % (gpu_card, batch_size)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 2:
        #TODO
        # if not input batch_size, we use batch_size=1
        batch_size = 1
        gpu_card = "p4"
        model = sys.argv[1]
        conf_name = "conf_%s" % model
        try:
            #write dead---no need from config
            #gpu_result_file = cf.get(conf_name, "gpu_result_filename")
            #top_result_file = cf.get(conf_name, "top_result_filename")
            gpu_result_file = "tensorrt_gpu_result_filename_%s.txt" % gpu_card
            top_result_file = "tensorrt_top_result_filename_%s.txt" % gpu_card
            #ut_yolo_path = cf.get(conf_name, "tensorrt_ut_yolo_path")
            ut_yolo_path = "/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output"
            jorcold_start_cmd = cf.get(conf_name, "tensorrt_jorcold_start_cmd") % (gpu_card, batch_size)
        except Exception as e:
            print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
            sys.exit(1)
    elif len(sys.argv) == 1:
        #write dead---no need from config
        #gpu_result_file = cf.get("conf_yolo", "gpu_result_filename")
        #top_result_file = cf.get("conf_yolo", "top_result_filename")
        # if not input batch_size, we use batch_size=1
        batch_size = 1
        gpu_card = "p4"
        model = "yolo"
        gpu_result_file = "tensorrt_gpu_result_filename_%s.txt" % gpu_card
        top_result_file = "tensorrt_top_result_filename_%s.txt" % gpu_card
        #ut_yolo_path = cf.get("conf_yolo", "tensorrt_ut_yolo_path")
        ut_yolo_path = "/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output"
        jorcold_start_cmd = cf.get("conf_yolo", "tensorrt_jorcold_start_cmd") % (gpu_card, batch_size)

    jorcold_start_test_yolo(gpu_result_file, top_result_file, ut_yolo_path, jorcold_start_cmd, model, gpu_card)

