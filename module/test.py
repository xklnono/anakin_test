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



if __name__ == '__main__':
    if len(sys.argv) == 3:
        model = sys.argv[1]
        batch_size = sys.argv[2]
        print "xxxxxxxxxxxxinput right, model: %s, batch_size: %sxxxxxxxxxxxx" % (model, batch_size)
        sys.exit(0)
    else:
        print "xxxxxxxxxxxxinput error!!!!xxxxxxxxxxxx"
        sys.exit(1)
