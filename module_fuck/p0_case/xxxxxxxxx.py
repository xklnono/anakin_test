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

if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    sys.exit(1)
