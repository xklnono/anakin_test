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
import ConfigParser

import mylogging


if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    #init config_parser
    try:
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/load_config.conf")
        if len(sys.argv) == 3:
            #TODO
            model = sys.argv[1]
            conf_name = "conf_%s" % model
            try:
                src_path = cf.get(conf_name, "src_path_tensorRT")
                dst_path = cf.get(conf_name, "dst_path_anakin2")
                name_list = cf.get(conf_name, "name_list")
            except Exception as e:
                print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
                sys.exit(1)
        elif len(sys.argv) == 2:
            #TODO
            model = sys.argv[1]
            conf_name = "conf_%s" % model
            try:
                src_path = cf.get(conf_name, "src_path_tensorRT")
                dst_path = cf.get(conf_name, "dst_path_anakin2")
                name_list = cf.get(conf_name, "name_list")
            except Exception as e:
                print ("\033[0;31;m[error]: Pls Check The Modle input wrong!\033[0m")
                sys.exit(1)
        elif len(sys.argv) == 1:
            src_path = cf.get("conf_yolo", "src_path_tensorRT")
            dst_path = cf.get("conf_yolo", "dst_path_anakin2")
            name_list = cf.get(conf_name, "name_list")
    except Exception as exception:
        print exception
        sys.exit(1)

    #check src_path and dst_path
    if not os.path.exists(src_path) or not os.path.exists(dst_path):
        print ("\033[0;31;m[error]: Pls Check The src or dst File Path!\nsrc file: %s\ndst file: %s\033[0m" % (src_path, dst_path))
        logging.error("NO_FILE_PATH: src(%s) vs dst(%s)" % (src_path, dst_path))
        sys.exit(0)
    
    for root, dirs, files in os.walk(src_path):
        for file in files:
            file_name_list = file.split('.', 1)
            if file_name_list[1] == 'jpg.txt':
                src_file = src_path + "/" + file
                dst_file = dst_path + "/" + file
                if os.path.exists(src_file) and os.path.exists(dst_file):
                    cmd = "python beyond_compared.py %s %s %s" % (src_file, dst_file, name_list)
                    print cmd
                    os.system(cmd)
                else:
                    print ("\033[0;31;44m[error]: No DST File Named: %s\033[0m" % (dst_file))
                    logging.error("NO_FILE: src(%s) vs dst(%s)" % (src_file, dst_file))

