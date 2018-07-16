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

import xlwt


if __name__ == '__main__':
    #init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet("anakin2.0 vs tensorRT")

    worksheet.write(0,0, label = 'this is test0')
    worksheet.write(1,0, label = 'this is test1')
    worksheet.write(2,0, label = 'this is test2')
    worksheet.write(3,0, label = 'this is test3')

    workbook.save('anakin_VS_RT.xls')

    sys.exit(1)
