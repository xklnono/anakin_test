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
Date:    2018/03/39 11:20:21
"""

import os
import re
import sys
import time
import json

names = ["cls_pred", "obj_pred", "ori_pred", "dim_pred"]
for i in range(len(names)):
    temp = names.pop(0)
    print temp

f = open("jingQC5007_12_1489483554_1489484255_6350.jpg.txt")

for line in f.readlines():
    pattern = "n = .*, c = .*, h = .*, w = .*"
    if re.match(pattern, line):
        print line

