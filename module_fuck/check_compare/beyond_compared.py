#!/usr/bin/python
# -*- coding=utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Compare TensorRT with anakin2.0 output Func() Method!

Authors: sysqa(sysqa@baidu.com)
Date:    2018/04/02
"""

import os
import re
import sys
import time
import json
import logging
import decimal
import ConfigParser
from decimal import Decimal

import mylogging

GLOBAL_SRC_DATA = "xxxxxxxx.txt"
GLOBAL_DST_DATA = "yyyyyyyyy.txt"

def tensorRT_read_inputfile_into_data(filename, name_list=""):
    #TODO
    if name_list == "":
        # use yolo model name_list default
        names = ["cls_pred", "obj_pred", "ori_pred", "dim_pred", "loc_pred"]
    else:
        # string to list
        names = name_list.split(",")
            
    f = open(filename)

    tensor = {}
    data_line = []
    for line in f.readlines():
        pattern = "n = .*, c = .*, h = .*, w = .*"
        if re.match(pattern, line):
            tensor_one = {}
            data = []
            
            temp_list = line.rstrip().split(",")
            key = {}
            #get key{}
            for item in temp_list:
                temp_list2 = item.strip().split("=")
                key[temp_list2[0].strip()] = int(temp_list2[1].strip())
            #put key{} into data{} then into tensor{}
            tensor_one["key"] = key
            tensor_one["data"] = data_line

            #print data_line
#            print tensor_one
#            print line

            tensor[names.pop(0)] = tensor_one

            #help to clean data_line
            data_line = []
        elif line != "\n":
            #each line is a hw list
            hw = []
            point_list = line.strip().rstrip(",").split(",")
            for item in point_list:
                hw.append(float(item.strip()))
            #print each line hw
            #print hw 
            data_line.append(hw)
        elif line == "\n":
            continue
          
    #print tensor
    f.close()
    return tensor
    
def compare_func_key(key1, key2, scope):
    if key1["n"] != key2["n"]:
        #print "========[ERROR]: IN tensor_%s, shape1(\"n\"):%d != shape2(\"n\"):%d" % (scope, key1["n"], key2["n"])
        logging.error("NO_RIGHT_SHAPE: IN tensor_%s, shape1(\"n\"):%d != shape2(\"n\"):%d" % (scope, key1["n"], key2["n"]))
        return False
    if key1["c"] != key2["c"]:
        #print "========[ERROR]: IN tensor_%s, shape1(\"c\"):%d != shape2(\"c\"):%d" % (scope, key1["c"], key2["c"])
        logging.error("NO_RIGHT_SHAPE: IN tensor_%s, shape1(\"c\"):%d != shape2(\"c\"):%d" % (scope, key1["c"], key2["c"]))
        return False
    if key1["h"] != key2["h"]:
        #print "========[ERROR]: IN tensor_%s, shape1(\"h\"):%d != shape2(\"h\"):%d" % (scope, key1["h"], key2["h"])
        logging.error("NO_RIGHT_SHAPE: IN tensor_%s, shape1(\"h\"):%d != shape2(\"h\"):%d" % (scope, key1["h"], key2["h"]))
        return False
    if key1["w"] != key2["w"]:
        #print "========[ERROR]: IN tensor_%s, shape1(\"w\"):%d != shape2(\"w\"):%d" % (scope, key1["w"], key2["w"])
        logging.error("NO_RIGHT_SHAPE: IN tensor_%s, shape1(\"w\"):%d != shape2(\"w\"):%d" % (scope, key1["w"], key2["w"]))
        return False
    return True

def compare_func_value(key1, key2, value1, value2, scope):
    #init decimal class
    context = decimal.getcontext()
    context.rounding = decimal.ROUND_05UP
    flag = True
    bingo = "nan"
    if len(value1) == len(value2):
        for i in range(len(value1)):
            line1 = value1[i]
            line2 = value2[i]
            if len(line1) == len(line2):
                for j in range(len(line1)):
                    channel = int(i / key1["h"]) + 1
                    height = (i % key1["h"]) + 1
                    width = j + 1
                    if line1[j] != line1[j] or line2[j] != line2[j]:
                        #print "--------[ERROR]: IN tensor_%s, value1(c=%d;h=%d;w=%d)(%f)=nan or value2(c=%d;h=%d;w=%d)(%f)=nan" % (scope, channel, height, width, line1[j], channel, height, width, line2[j]) 
                        logging.error("NO_DATA: IN tensor_%s, value1(c=%d;h=%d;w=%d)(%f)=nan or value2(c=%d;h=%d;w=%d)(%f)=nan" % (scope, channel, height, width, line1[j], channel, height, width, line2[j]))
                        flag = False
                        return flag
                    float_line1 = float(line1[j])
                    float_line2 = float(line2[j])

                    beyond_a = abs(float_line1)
                    beyond_b = abs(float_line2)
                    d_value = abs(float_line1 - float_line2)

                    if min(beyond_a, beyond_b) == 0:
                        relative_error = 0
                    else:
                        relative_error = d_value / min(beyond_a, beyond_b)

                    if beyond_a < 1:
                        king_beyond = d_value
                    else:
                        king_beyond = relative_error

                    if king_beyond >= 0.001:
                        #print "--------[ERROR]: IN tensor_%s, value1(c=%d;h=%d;w=%d)(%f) != value2(c=%d;h=%d;w=%d)(%f), d_value: %s" % (scope, channel, height, width, line1[j], channel, height, width, line2[j], king_beyond) 
                        logging.error("NO_RIGHT_DATA: IN tensor_%s, value1(c=%d;h=%d;w=%d)(%f) != value2(c=%d;h=%d;w=%d)(%f), d_value: %s" % (scope, channel, height, width, line1[j], channel, height, width, line2[j], king_beyond))
                        flag = False
                        return flag
            else:
                #print "--------[ERROR]: IN tensor_%s, 1_line_num(w):%d != 2_line_num(w):%d" % (scope, key1["w"], key2["w"])
                logging.error("NO_RIGHT_WEIGHT: IN tensor_%s, 1_line_num(w):%d != 2_line_num(w):%d" % (scope, key1["w"], key2["w"]))
                return False
    else:
        #len() = c * h
        #print "--------[ERROR]: IN tensor_%s, 1_all_line_num(c*h):%d*%d != 2_all_line_num(c*h):%d*%d" % (scope, key1["c"], key1["h"], key2["c"], key2["h"])
        logging.error("NO_RIGHT_C_AND_H: IN tensor_%s, 1_all_line_num(c*h):%d*%d != 2_all_line_num(c*h):%d*%d" % (scope, key1["c"], key1["h"], key2["c"], key2["h"]))
        return False
#    print "\n"
    return flag
    
def compare_dataA_with_dataB(src_data, dst_data):
    flag_all = True

    for name_list in src_data.keys():
        if dst_data.has_key(name_list):
            #compare shape1 ?= shape2
            shape1 = src_data[name_list]["key"]
            shape2 = dst_data[name_list]["key"]
            flag = compare_func_key(shape1, shape2, name_list)

            #compare value1 ?= value2
            if flag == True:
                value1 = src_data[name_list]["data"]
                value2 = dst_data[name_list]["data"]
                if not compare_func_value(shape1, shape2, value1, value2, name_list):
                    flag_all = False
            else:
                flag_all = False
        else:
            #print "========[ERROR]: Anakin2_file do not have name_list:%s" % (name_list)
            logging.error("NO_NAME_LIST: Anakin2_file do not have name_list:%s" % (name_list))
            flag_all = False

    return flag_all


if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    if len(sys.argv) == 1:
        src_file = GLOBAL_SRC_DATA
        dst_file = GLOBAL_DST_DATA
    elif len(sys.argv) == 3:
        #judge file is exist
        src_file = sys.argv[1]
        dst_file = sys.argv[2]
    elif len(sys.argv) == 4:
        #judge file is exist
        src_file = sys.argv[1]
        dst_file = sys.argv[2]
        name_list = sys.argv[3]
    elif len(sys.argv) == 2:
        file = sys.argv[1]
        # use yolo model default
        tensorRT_read_inputfile_into_data(file)
        sys.exit(0)

    if not os.path.exists(src_file):
        print "src_file is not exist: %s" % src_file
    if not os.path.exists(dst_file):
        print "dst_file is not exist: %s" % dst_file

    src_data = tensorRT_read_inputfile_into_data(src_file, name_list)
    dst_data = tensorRT_read_inputfile_into_data(dst_file, name_list)

    flag = compare_dataA_with_dataB(src_data, dst_data)
    if not flag:
        print ("\033[0;31;mNO_PASS: src(%s) vs dst(%s)\033[0m" % (src_file, dst_file))
        logging.error("NO_PASS: src(%s) vs dst(%s)" % (src_file, dst_file))
    else:
        print ("\033[0;36;mPASS: src(%s) vs dst(%s)\033[0m" % (src_file, dst_file))
        logging.info("PASS: src(%s) vs dst(%s)" % (src_file, dst_file))
        
        
