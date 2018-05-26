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
#import MysqlHelper as mysql_helper


if __name__ == '__main__':
    if len(sys.argv) == 3:
        model = sys.argv[1]
        batch_size = sys.argv[2]
    else:
        print "xxxxxxxxxxxxinput error!!!!xxxxxxxxxxxx"
