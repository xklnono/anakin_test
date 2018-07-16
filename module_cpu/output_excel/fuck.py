#!/usr/bin/python
# -*- coding=utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Compare paddle/lego with anakin2.0 output Func() Method!

Authors: sysqa(sysqa@baidu.com)
Date:    2018/06/02
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
from load_common import LoadCommon

import xlwt

GLOBAL_LINE_5117 = 1
GLOBAL_LINE_v3 = 1
GLOBAL_LINE_v4 = 1

class LoadPerformance(object):
    """
    init
    """
    def __init__(self, db_name, model, thread_size):
        """
        init
        """
        # init mysql
        self.mysql = LoadCommon(db_name)
        self.thread_size = thread_size
        self.model = model

    def make_excel_result(self):
        """
        analysis the excel data
        1. Net_name 
        2. thread_size
        3. Library: Anakin2
        4. Anakin2 Latency (ms)
        5. Anakin2 QPS (ms)
        6. Anakin2 ratio (%)
        7. Anakin2 mem (%)
        8. Library: Paddle
        9. Paddle Latency (ms)
        10. Paddle QPS (ms)
        11. Paddle ratio (%)
        12. Paddle mem (%)
        13. anakin/paddle latency
        14. anakin/paddle qps
        15. accuracy rate
        """
        # 1.Net_name
        net_name = self.model
        # 2. thread_size
        thread_size = self.thread_size
        # 3. Library: Anakin2
        Library_anakin = "Anakin2"
        # 8. Library: paddle
        Library_paddle = "Paddle"
        Library_lego = "Lego"

        # 4. anakin2 Latency (ms)
        anakin_latency_5117 = self.mysql.select_anakin2_latency("5117")
        anakin_latency_v3 = self.mysql.select_anakin2_latency("v3")
        anakin_latency_v4 = self.mysql.select_anakin2_latency("v4")
        print "anakin_latency_5117: %s" % anakin_latency_5117
        print "anakin_latency_v3: %s" % anakin_latency_v3
        print "anakin_latency_v4: %s" % anakin_latency_v4
        # 5. anakin2 QPS (ms)
        anakin_qps_5117 = self.mysql.select_anakin2_qps("5117")
        anakin_qps_v3 = self.mysql.select_anakin2_qps("v3")
        anakin_qps_v4 = self.mysql.select_anakin2_qps("v4")
        print "anakin_qps_5117: %s" % anakin_qps_5117
        print "anakin_qps_v3: %s" % anakin_qps_v3
        print "anakin_qps_v4: %s" % anakin_qps_v4
        # 6. anakin2 ratio (ms)
        anakin_ratio_5117 = self.mysql.select_anakin2_ratio("5117")
        anakin_ratio_v3 = self.mysql.select_anakin2_ratio("v3")
        anakin_ratio_v4 = self.mysql.select_anakin2_ratio("v4")
        print "anakin_ratio_5117: %s" % anakin_ratio_5117
        print "anakin_ratio_v3: %s" % anakin_ratio_v3
        print "anakin_ratio_v4: %s" % anakin_ratio_v4
        # 7. anakin2 Mem (MB)
        anakin_memory_5117  = self.mysql.select_anakin2_memory("5117")
        anakin_memory_v3  = self.mysql.select_anakin2_memory("v3")
        anakin_memory_v4  = self.mysql.select_anakin2_memory("v4")
        print "anakin_memory_5117: %s" % anakin_memory_5117
        print "anakin_memory_v3: %s" % anakin_memory_v3
        print "anakin_memory_v4: %s" % anakin_memory_v4

        # 9. paddle Latency (ms)
        paddle_latency_5117 = self.mysql.select_paddle_latency("5117")
        paddle_latency_v3 = self.mysql.select_paddle_latency("v3")
        paddle_latency_v4 = self.mysql.select_paddle_latency("v4")
        print "paddle_latency_5117: %s" % paddle_latency_5117
        print "paddle_latency_v3: %s" % paddle_latency_v3
        print "paddle_latency_v4: %s" % paddle_latency_v4
        # 10. paddle QPS (ms)
        paddle_qps_5117 = self.mysql.select_paddle_qps("5117")
        paddle_qps_v3 = self.mysql.select_paddle_qps("v3")
        paddle_qps_v4 = self.mysql.select_paddle_qps("v4")
        print "paddle_qps_5117: %s" % paddle_qps_5117
        print "paddle_qps_v3: %s" % paddle_qps_v3
        print "paddle_qps_v4: %s" % paddle_qps_v4
        # 11. paddle ratio (ms)
        paddle_ratio_5117 = self.mysql.select_paddle_ratio("5117")
        paddle_ratio_v3 = self.mysql.select_paddle_ratio("v3")
        paddle_ratio_v4 = self.mysql.select_paddle_ratio("v4")
        print "paddle_ratio_5117: %s" % paddle_ratio_5117
        print "paddle_ratio_v3: %s" % paddle_ratio_v3
        print "paddle_ratio_v4: %s" % paddle_ratio_v4
        # 12. paddle Mem (MB)
        paddle_memory_5117  = self.mysql.select_paddle_memory("5117")
        paddle_memory_v3  = self.mysql.select_paddle_memory("v3")
        paddle_memory_v4  = self.mysql.select_paddle_memory("v4")
        print "paddle_memory_5117: %s" % paddle_memory_5117
        print "paddle_memory_v3: %s" % paddle_memory_v3
        print "paddle_memory_v4: %s" % paddle_memory_v4

        # 9. lego Latency (ms)
        lego_latency_5117 = self.mysql.select_lego_latency("5117")
        lego_latency_v3 = self.mysql.select_lego_latency("v3")
        lego_latency_v4 = self.mysql.select_lego_latency("v4")
        print "lego_latency_5117: %s" % lego_latency_5117
        print "lego_latency_v3: %s" % lego_latency_v3
        print "lego_latency_v4: %s" % lego_latency_v4
        # 10. lego QPS (ms)
        lego_qps_5117 = self.mysql.select_lego_qps("5117")
        lego_qps_v3 = self.mysql.select_lego_qps("v3")
        lego_qps_v4 = self.mysql.select_lego_qps("v4")
        print "lego_qps_5117: %s" % lego_qps_5117
        print "lego_qps_v3: %s" % lego_qps_v3
        print "lego_qps_v4: %s" % lego_qps_v4
        # 11. lego ratio (ms)
        lego_ratio_5117 = self.mysql.select_lego_ratio("5117")
        lego_ratio_v3 = self.mysql.select_lego_ratio("v3")
        lego_ratio_v4 = self.mysql.select_lego_ratio("v4")
        print "lego_ratio_5117: %s" % lego_ratio_5117
        print "lego_ratio_v3: %s" % lego_ratio_v3
        print "lego_ratio_v4: %s" % lego_ratio_v4
        # 12. lego Mem (MB)
        lego_memory_5117  = self.mysql.select_lego_memory("5117")
        lego_memory_v3  = self.mysql.select_lego_memory("v3")
        lego_memory_v4  = self.mysql.select_lego_memory("v4")
        print "lego_memory_5117: %s" % lego_memory_5117
        print "lego_memory_v3: %s" % lego_memory_v3
        print "lego_memory_v4: %s" % lego_memory_v4

        # 13. anakin/paddle latency
        if anakin_latency_5117 and paddle_latency_5117:
            ratio_latency_5117 = str(int((float(anakin_latency_5117) / float(paddle_latency_5117)) * 100)) + "%"
        else:
            ratio_latency_5117 = None
        if anakin_latency_v3 and paddle_latency_v3:
            ratio_latency_v3 = str(int((float(anakin_latency_v3) / float(paddle_latency_v3)) * 100)) + "%"
        else:
            ratio_latency_v3 = None
        if anakin_latency_v4 and paddle_latency_v4:
            ratio_latency_v4 = str(int((float(anakin_latency_v4) / float(paddle_latency_v4)) * 100)) + "%"
        else:
            ratio_latency_v4 = None
        print "ratio_latency_5117: %s" % ratio_latency_5117
        print "ratio_latency_v3: %s" % ratio_latency_v3
        print "ratio_latency_v4: %s" % ratio_latency_v4

        # 14. anakin/paddle qps
        if anakin_qps_5117 and paddle_qps_5117:
            ratio_qps_5117 = str(int((float(paddle_qps_5117) / float(anakin_qps_5117)) * 100)) + "%"
        else:
            ratio_qps_5117 = None

        if anakin_qps_v3 and paddle_qps_v3:
            ratio_qps_v3 = str(int((float(paddle_qps_v3) / float(anakin_qps_v3)) * 100)) + "%"
        else:
            ratio_qps_v3 = None

        if anakin_qps_v4 and paddle_qps_v4:
            ratio_qps_v4 = str(int((float(paddle_qps_v4) / float(anakin_qps_v4)) * 100)) + "%"
        else:
            ratio_qps_v4 = None
        print "ratio_qps_5117: %s" % ratio_qps_5117
        print "ratio_qps_v3: %s" % ratio_qps_v3
        print "ratio_qps_v4: %s" % ratio_qps_v4

        # 13. anakin/lego latency
        if anakin_latency_5117 and lego_latency_5117:
            ratio_latency_5117_2 = str(int((float(anakin_latency_5117) / float(lego_latency_5117)) * 100)) + "%"
        else:
            ratio_latency_5117_2 = None
        if anakin_latency_v3 and lego_latency_v3:
            ratio_latency_v3_2 = str(int((float(anakin_latency_v3) / float(lego_latency_v3)) * 100)) + "%"
        else:
            ratio_latency_v3_2 = None
        if anakin_latency_v4 and lego_latency_v4:
            ratio_latency_v4_2 = str(int((float(anakin_latency_v4) / float(lego_latency_v4)) * 100)) + "%"
        else:
            ratio_latency_v4_2 = None
        print "ratio_latency_5117_2: %s" % ratio_latency_5117_2
        print "ratio_latency_v3_2: %s" % ratio_latency_v3_2
        print "ratio_latency_v4_2: %s" % ratio_latency_v4_2

        # 14. anakin/lego qps
        if anakin_qps_5117 and lego_qps_5117:
            ratio_qps_5117_2 = str(int((float(lego_qps_5117) / float(anakin_qps_5117)) * 100)) + "%"
        else:
            ratio_qps_5117_2 = None

        if anakin_qps_v3 and lego_qps_v3:
            ratio_qps_v3_2 = str(int((float(lego_qps_v3) / float(anakin_qps_v3)) * 100)) + "%"
        else:
            ratio_qps_v3_2 = None

        if anakin_qps_v4 and lego_qps_v4:
            ratio_qps_v4_2 = str(int((float(lego_qps_v4) / float(anakin_qps_v4)) * 100)) + "%"
        else:
            ratio_qps_v4_2 = None
        print "ratio_qps_5117_2: %s" % ratio_qps_5117_2
        print "ratio_qps_v3_2: %s" % ratio_qps_v3_2
        print "ratio_qps_v4_2: %s" % ratio_qps_v4_2


#        # 15. accuracy rate
#        accuracy_rate_5117 = self.mysql.select_accuracy_rate("5117")
#        accuracy_rate_v3 = self.mysql.select_accuracy_rate("v3")
#        accuracy_rate_v4 = self.mysql.select_accuracy_rate("v4")
#        if accuracy_rate_5117 == 0.0:
#            ratio_accuracy_rate_5117 = "0%"
#        elif accuracy_rate_5117 and accuracy_rate_5117 != 0.0:
#            ratio_accuracy_rate_5117 = str(int(accuracy_rate_5117) * 100) + "%"
#        else:
#            ratio_accuracy_rate_5117 = None
#
#        if accuracy_rate_v3 == 0.0:
#            ratio_accuracy_rate_v3 = "0%"
#        elif accuracy_rate_v3 and accuracy_rate_v3 != 0.0:
#            ratio_accuracy_rate_v3 = str(int(accuracy_rate_v3) * 100) + "%"
#        else:
#            ratio_accuracy_rate_v3 = None

#        if accuracy_rate_v4 == 0.0:
#            ratio_accuracy_rate_v4 = "0%"
#        elif accuracy_rate_v4 and accuracy_rate_v4 != 0.0:
#            ratio_accuracy_rate_v4 = str(int(accuracy_rate_v4) * 100) + "%"
#        else:
#            ratio_accuracy_rate_v4 = None
#
#        print "ratio_accuracy_rate_5117: %s" % ratio_accuracy_rate_5117
#        print "ratio_accuracy_rate_v4: %s" % ratio_accuracy_rate_v4
      
        line_data_5117 = {}
        line_data_5117["net_name"] = net_name
        line_data_5117["thread_size"] = thread_size
        line_data_5117["Library_anakin"] = Library_anakin + "_5117"
        line_data_5117["anakin_latency_5117"] = anakin_latency_5117
        line_data_5117["anakin_qps_5117"] = anakin_qps_5117
        line_data_5117["anakin_ratio_5117"] = anakin_ratio_5117
        line_data_5117["anakin_memory_5117"] = anakin_memory_5117

        line_data_5117["Library_paddle"] = Library_paddle + "_5117"
        line_data_5117["paddle_latency_5117"] = paddle_latency_5117
        line_data_5117["paddle_qps_5117"] = paddle_qps_5117
        line_data_5117["paddle_ratio_5117"] = paddle_ratio_5117
        line_data_5117["paddle_memory_5117"] = paddle_memory_5117

        line_data_5117["Library_lego"] = Library_lego + "_5117"
        line_data_5117["lego_latency_5117"] = lego_latency_5117
        line_data_5117["lego_qps_5117"] = lego_qps_5117
        line_data_5117["lego_ratio_5117"] = lego_ratio_5117
        line_data_5117["lego_memory_5117"] = lego_memory_5117

        line_data_5117["ratio_latency_5117"] = ratio_latency_5117
        line_data_5117["ratio_qps_5117"] = ratio_qps_5117
        line_data_5117["ratio_latency_5117_2"] = ratio_latency_5117_2
        line_data_5117["ratio_qps_5117_2"] = ratio_qps_5117_2
#        line_data_5117["ratio_accuracy_rate_5117"] = ratio_accuracy_rate_5117

        line_data_v3 = {}
        line_data_v3["net_name"] = net_name
        line_data_v3["thread_size"] = thread_size

        line_data_v3["Library_anakin"] = Library_anakin + "_v3"
        line_data_v3["anakin_latency_v3"] = anakin_latency_v3
        line_data_v3["anakin_qps_v3"] = anakin_qps_v3
        line_data_v3["anakin_ratio_v3"] = anakin_ratio_v3
        line_data_v3["anakin_memory_v3"] = anakin_memory_v3

        line_data_v3["Library_paddle"] = Library_paddle + "_v3"
        line_data_v3["paddle_latency_v3"] = paddle_latency_v3
        line_data_v3["paddle_qps_v3"] = paddle_qps_v3
        line_data_v3["paddle_ratio_v3"] = paddle_ratio_v3
        line_data_v3["paddle_memory_v3"] = paddle_memory_v3

        line_data_v3["Library_lego"] = Library_lego + "_v3"
        line_data_v3["lego_latency_v3"] = lego_latency_v3
        line_data_v3["lego_qps_v3"] = lego_qps_v3
        line_data_v3["lego_ratio_v3"] = lego_ratio_v3
        line_data_v3["lego_memory_v3"] = lego_memory_v3

        line_data_v3["ratio_latency_v3"] = ratio_latency_v3
        line_data_v3["ratio_qps_v3"] = ratio_qps_v3
        line_data_v3["ratio_latency_v3_2"] = ratio_latency_v3_2
        line_data_v3["ratio_qps_v3_2"] = ratio_qps_v3_2
#        line_data_v3["ratio_accuracy_rate_v3"] = ratio_accuracy_rate_v3

        line_data_v4 = {}
        line_data_v4["net_name"] = net_name
        line_data_v4["thread_size"] = thread_size

        line_data_v4["Library_anakin"] = Library_anakin + "_v4"
        line_data_v4["anakin_latency_v4"] = anakin_latency_v4
        line_data_v4["anakin_qps_v4"] = anakin_qps_v4
        line_data_v4["anakin_ratio_v4"] = anakin_ratio_v4
        line_data_v4["anakin_memory_v4"] = anakin_memory_v4

        line_data_v4["Library_paddle"] = Library_paddle + "_v4"
        line_data_v4["paddle_latency_v4"] = paddle_latency_v4
        line_data_v4["paddle_qps_v4"] = paddle_qps_v4
        line_data_v4["paddle_ratio_v4"] = paddle_ratio_v4
        line_data_v4["paddle_memory_v4"] = paddle_memory_v4

        line_data_v4["Library_lego"] = Library_lego + "_v4"
        line_data_v4["lego_latency_v4"] = lego_latency_v4
        line_data_v4["lego_qps_v4"] = lego_qps_v4
        line_data_v4["lego_ratio_v4"] = lego_ratio_v4
        line_data_v4["lego_memory_v4"] = lego_memory_v4

        line_data_v4["ratio_latency_v4"] = ratio_latency_v4
        line_data_v4["ratio_qps_v4"] = ratio_qps_v4
        line_data_v4["ratio_latency_v4_2"] = ratio_latency_v4_2
        line_data_v4["ratio_qps_v4_2"] = ratio_qps_v4_2
#        line_data_v4["ratio_accuracy_rate_v4"] = ratio_accuracy_rate_v4

        return line_data_5117, line_data_v3, line_data_v4

       
class LoadExcel(object):
    """
    init
    """
    def __init__(self):
        """
        init
        """
        # init excel
        self.workbook = xlwt.Workbook(encoding = 'utf-8')
        self.worksheet_5117 = self.workbook.add_sheet("5117")
        self.worksheet_v3 = self.workbook.add_sheet("v3")
        self.worksheet_v4 = self.workbook.add_sheet("v4")

        # set excel style
        alignment = xlwt.Alignment() # Create Alignment
        alignment.horz = xlwt.Alignment.HORZ_CENTER
        alignment.vert = xlwt.Alignment.VERT_CENTER
        alignment.wrap = xlwt.Alignment.WRAP_AT_RIGHT

        # word is black
        self.style = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style.font = font
        self.style.alignment = alignment

        # word is red
        self.style_red = xlwt.XFStyle()
        font_red = xlwt.Font()
        font_red.name = 'PT Mono'
        font_red.height = 240
        font_red.colour_index = 2
        self.style_red.font = font_red
        self.style_red.alignment = alignment

        # background is color
        col = self.worksheet_5117.col(0)
        col.width = 256 * 25
        col1 = self.worksheet_v3.col(0)
        col1.width = 256 * 25
        col2 = self.worksheet_v4.col(0)
        col2.width = 256 * 25
        self.style_background0 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background0.font = font
        self.style_background0.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 22
        self.style_background0.pattern = pattern
        self.worksheet_5117.write(0, 0, "Net_Name", self.style_background0)
        self.worksheet_v3.write(0, 0, "Net_Name", self.style_background0)
        self.worksheet_v4.write(0, 0, "Net_Name", self.style_background0)

        col = self.worksheet_5117.col(1)
        col.width = 256 * 15
        col1 = self.worksheet_v3.col(1)
        col1.width = 256 * 15
        col2 = self.worksheet_v4.col(1)
        col2.width = 256 * 15
        self.style_background1 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background1.font = font
        self.style_background1.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 26
        self.style_background1.pattern = pattern
        self.worksheet_5117.write(0, 1, "Thread_Size", self.style_background1)
        self.worksheet_v3.write(0, 1, "Thread_Size", self.style_background1)
        self.worksheet_v4.write(0, 1, "Thread_Size", self.style_background1)

        col = self.worksheet_5117.col(2)
        col.width = 256 * 15
        col1 = self.worksheet_v3.col(2)
        col1.width = 256 * 15
        col2 = self.worksheet_v4.col(2)
        col2.width = 256 * 15
        self.style_background2 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background2.font = font
        self.style_background2.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 48
        self.style_background2.pattern = pattern
        self.worksheet_5117.write(0, 2, "Library", self.style_background2)
        self.worksheet_v3.write(0, 2, "Library", self.style_background2)
        self.worksheet_v4.write(0, 2, "Library", self.style_background2)

        col = self.worksheet_5117.col(3)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(3)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(3)
        col2.width = 256 * 20
        self.style_background3 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background3.font = font
        self.style_background3.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 42
        self.style_background3.pattern = pattern
        self.worksheet_5117.write(0, 3, "Latency (ms)", self.style_background3)
        self.worksheet_v3.write(0, 3, "Latency (ms)", self.style_background3)
        self.worksheet_v4.write(0, 3, "Latency (ms)", self.style_background3)

        col = self.worksheet_5117.col(4)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(4)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(4)
        col2.width = 256 * 20
        self.style_background4 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background4.font = font
        self.style_background4.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 43
        self.style_background4.pattern = pattern
        self.worksheet_5117.write(0, 4, "QPS", self.style_background4)
        self.worksheet_v3.write(0, 4, "QPS", self.style_background4)
        self.worksheet_v4.write(0, 4, "QPS", self.style_background4)

        col = self.worksheet_5117.col(5)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(5)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(5)
        col2.width = 256 * 20
        self.style_background5 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background5.font = font
        self.style_background5.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 47
        self.style_background5.pattern = pattern
        self.worksheet_5117.write(0, 5, "CPU Ratio (%)", self.style_background5)
        self.worksheet_v3.write(0, 5, "CPU Ratio (%)", self.style_background5)
        self.worksheet_v4.write(0, 5, "CPU Ratio (%)", self.style_background5)

        col = self.worksheet_5117.col(6)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(6)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(6)
        col2.width = 256 * 20
        self.style_background6 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background6.font = font
        self.style_background6.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 44
        self.style_background6.pattern = pattern
        self.worksheet_5117.write(0, 6, "Memory (G)", self.style_background6)
        self.worksheet_v3.write(0, 6, "Memory (G)", self.style_background6)
        self.worksheet_v4.write(0, 6, "Memory (G)", self.style_background6)

        col = self.worksheet_5117.col(7)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(7)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(7)
        col2.width = 256 * 20
        self.style_background7 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background7.font = font
        self.style_background7.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 53
        self.style_background7.pattern = pattern
        self.worksheet_5117.write(0, 7, "anakin/paddle\nlatency", self.style_background7)
        self.worksheet_v3.write(0, 7, "anakin/paddle\nlatency", self.style_background7)
        self.worksheet_v4.write(0, 7, "anakin/paddle\nlatency", self.style_background7)

        col = self.worksheet_5117.col(8)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(8)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(8)
        col2.width = 256 * 20
        self.style_background8 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background8.font = font
        self.style_background8.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 53
        self.style_background8.pattern = pattern
        self.worksheet_5117.write(0, 8, "anakin/paddle\nqps", self.style_background8)
        self.worksheet_v3.write(0, 8, "anakin/paddle\nqps", self.style_background8)
        self.worksheet_v4.write(0, 8, "anakin/paddle\nqps", self.style_background8)

        col = self.worksheet_5117.col(9)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(9)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(9)
        col2.width = 256 * 20
        self.style_background9 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background9.font = font
        self.style_background9.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 54
        self.style_background9.pattern = pattern
        self.worksheet_5117.write(0, 9, "anakin/lego\nlatency", self.style_background9)
        self.worksheet_v3.write(0, 9, "anakin/lego\nlatency", self.style_background9)
        self.worksheet_v4.write(0, 9, "anakin/lego\nlatency", self.style_background9)

        col = self.worksheet_5117.col(10)
        col.width = 256 * 20
        col1 = self.worksheet_v3.col(10)
        col1.width = 256 * 20
        col2 = self.worksheet_v4.col(10)
        col2.width = 256 * 20
        self.style_background10 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background10.font = font
        self.style_background10.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 54
        self.style_background10.pattern = pattern
        self.worksheet_5117.write(0, 10, "anakin/lego\nqps", self.style_background10)
        self.worksheet_v3.write(0, 10, "anakin/lego\nqps", self.style_background10)
        self.worksheet_v4.write(0, 10, "anakin/lego\nqps", self.style_background10)

#        col = self.worksheet_5117.col(14)
#        col.width = 256 * 20
#        col1 = self.worksheet_v3.col(14)
#        col1.width = 256 * 20
#        col2 = self.worksheet_v4.col(14)
#        col2.width = 256 * 20
#        self.style_background14 = xlwt.XFStyle() # Create Style
#        font = xlwt.Font()
#        font.name = 'PT Mono'
#        font.height = 240
#        font.colour_index = 0
#        self.style_background14.font = font
#        self.style_background14.alignment = alignment
#        pattern = xlwt.Pattern()
#        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
#        pattern.pattern_fore_colour = 53
#        self.style_background14.pattern = pattern
#        self.worksheet_5117.write(0, 14, "accuracy rate\n(%)", self.style_background14)
#        self.worksheet_v3.write(0, 14, "accuracy rate\n(%)", self.style_background14)
#        self.worksheet_v4.write(0, 14, "accuracy rate\n(%)", self.style_background14)

    def __del__(self):
        """
        del
        """
        # save the excel 
        self.workbook.save('anakin_VS_RT.xls')

    def put_in_excel_5117(self, line_data):
        """
        input excel
        """
        global GLOBAL_LINE_5117
        # input excel
        # 1. 5117 sheet
        self.worksheet_5117.write(GLOBAL_LINE_5117, 0, line_data["net_name"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 1, line_data["thread_size"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 2, line_data["Library_anakin"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 3, line_data["anakin_latency_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 4, line_data["anakin_qps_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 5, line_data["anakin_ratio_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 6, line_data["anakin_memory_5117"], self.style)

        if line_data["ratio_latency_5117"]:
            data = int(line_data["ratio_latency_5117"].rstrip("%"))
            if data >= 100:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 7, line_data["ratio_latency_5117"], self.style_red)
            else:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 7, line_data["ratio_latency_5117"], self.style)
        else:
            self.worksheet_5117.write(GLOBAL_LINE_5117, 7, line_data["ratio_latency_5117"], self.style)
     
        if line_data["ratio_qps_5117"]:
            data = int(line_data["ratio_qps_5117"].rstrip("%"))
            if data >= 100:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 8, line_data["ratio_qps_5117"], self.style_red)
            else:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 8, line_data["ratio_qps_5117"], self.style)
        else:
            self.worksheet_5117.write(GLOBAL_LINE_5117, 8, line_data["ratio_qps_5117"], self.style)

        if line_data["ratio_latency_5117_2"]:
            data = int(line_data["ratio_latency_5117_2"].rstrip("%"))
            if data >= 100:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 9, line_data["ratio_latency_5117_2"], self.style_red)
            else:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 9, line_data["ratio_latency_5117_2"], self.style)
        else:
            self.worksheet_5117.write(GLOBAL_LINE_5117, 9, line_data["ratio_latency_5117_2"], self.style)

        if line_data["ratio_qps_5117_2"]:
            data = int(line_data["ratio_qps_5117_2"].rstrip("%"))
            if data >= 100:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 10, line_data["ratio_qps_5117_2"], self.style_red)
            else:
                self.worksheet_5117.write(GLOBAL_LINE_5117, 10, line_data["ratio_qps_5117_2"], self.style)
        else:
            self.worksheet_5117.write(GLOBAL_LINE_5117, 10, line_data["ratio_qps_5117_2"], self.style)

#        if line_data["ratio_accuracy_rate_5117"]:
#            data = int(line_data["ratio_accuracy_rate_5117"].rstrip("%"))
#            if data == 100:
#                self.worksheet_5117.write(GLOBAL_LINE_5117, 14, line_data["ratio_accuracy_rate_5117"], self.style)
#            else:
#                self.worksheet_5117.write(GLOBAL_LINE_5117, 14, line_data["ratio_accuracy_rate_5117"], self.style_red)
#        else:
#            self.worksheet_5117.write(GLOBAL_LINE_5117, 14, line_data["ratio_accuracy_rate_5117"], self.style)

        GLOBAL_LINE_5117 += 1
        self.worksheet_5117.write(GLOBAL_LINE_5117, 2, line_data["Library_paddle"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 3, line_data["paddle_latency_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 4, line_data["paddle_qps_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 5, line_data["paddle_ratio_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 6, line_data["paddle_memory_5117"], self.style)

        GLOBAL_LINE_5117 += 1
        self.worksheet_5117.write(GLOBAL_LINE_5117, 2, line_data["Library_lego"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 3, line_data["lego_latency_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 4, line_data["lego_qps_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 5, line_data["lego_ratio_5117"], self.style)
        self.worksheet_5117.write(GLOBAL_LINE_5117, 6, line_data["lego_memory_5117"], self.style)

    def put_in_excel_v3(self, line_data):
        """
        input excel
        """
        global GLOBAL_LINE_v3
        # input excel
        # 2. v3 sheet
        self.worksheet_v3.write(GLOBAL_LINE_v3, 0, line_data["net_name"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 1, line_data["thread_size"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 2, line_data["Library_anakin"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 3, line_data["anakin_latency_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 4, line_data["anakin_qps_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 5, line_data["anakin_ratio_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 6, line_data["anakin_memory_v3"], self.style)

        if line_data["ratio_latency_v3"]:
            data = int(line_data["ratio_latency_v3"].rstrip("%"))
            if data >= 100:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 7, line_data["ratio_latency_v3"], self.style_red)
            else:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 7, line_data["ratio_latency_v3"], self.style)
        else:
            self.worksheet_v3.write(GLOBAL_LINE_v3, 7, line_data["ratio_latency_v3"], self.style)

        if line_data["ratio_qps_v3"]:
            data = int(line_data["ratio_qps_v3"].rstrip("%"))
            if data >= 100:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 8, line_data["ratio_qps_v3"], self.style_red)
            else:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 8, line_data["ratio_qps_v3"], self.style)
        else:
            self.worksheet_v3.write(GLOBAL_LINE_v3, 8, line_data["ratio_qps_v3"], self.style) 

        if line_data["ratio_latency_v3_2"]:
            data = int(line_data["ratio_latency_v3_2"].rstrip("%"))
            if data >= 100:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 9, line_data["ratio_latency_v3_2"], self.style_red)
            else:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 9, line_data["ratio_latency_v3_2"], self.style)
        else:
            self.worksheet_v3.write(GLOBAL_LINE_v3, 9, line_data["ratio_latency_v3_2"], self.style)

        if line_data["ratio_qps_v3_2"]:
            data = int(line_data["ratio_qps_v3_2"].rstrip("%"))
            if data >= 100:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 10, line_data["ratio_qps_v3_2"], self.style_red)
            else:
                self.worksheet_v3.write(GLOBAL_LINE_v3, 10, line_data["ratio_qps_v3_2"], self.style)
        else:
            self.worksheet_v3.write(GLOBAL_LINE_v3, 10, line_data["ratio_qps_v3_2"], self.style)

#        if line_data["ratio_accuracy_rate_v3"]:
#            data = int(line_data["ratio_accuracy_rate_v3"].rstrip("%"))
#            if data == 100:
#                self.worksheet_v3.write(GLOBAL_LINE_v3, 10, line_data["ratio_accuracy_rate_v3"], self.style)
#            else:
#                self.worksheet_v3.write(GLOBAL_LINE_v3, 10, line_data["ratio_accuracy_rate_v3"], self.style_red)
#        else:
#            self.worksheet_v3.write(GLOBAL_LINE_v3, 10, line_data["ratio_accuracy_rate_v3"], self.style)

        GLOBAL_LINE_v3 += 1
        self.worksheet_v3.write(GLOBAL_LINE_v3, 2, line_data["Library_paddle"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 3, line_data["paddle_latency_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 4, line_data["paddle_qps_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 5, line_data["paddle_ratio_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 6, line_data["paddle_memory_v3"], self.style)

        GLOBAL_LINE_v3 += 1
        self.worksheet_v3.write(GLOBAL_LINE_v3, 2, line_data["Library_lego"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 3, line_data["lego_latency_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 4, line_data["lego_qps_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 5, line_data["lego_ratio_v3"], self.style)
        self.worksheet_v3.write(GLOBAL_LINE_v3, 6, line_data["lego_memory_v3"], self.style)

    def put_in_excel_v4(self, line_data):
        """
        input excel
        """
        global GLOBAL_LINE_v4
        # input excel
        # 2. v4 sheet
        self.worksheet_v4.write(GLOBAL_LINE_v4, 0, line_data["net_name"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 1, line_data["thread_size"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 2, line_data["Library_anakin"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 3, line_data["anakin_latency_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 4, line_data["anakin_qps_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 5, line_data["anakin_ratio_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 6, line_data["anakin_memory_v4"], self.style)

        if line_data["ratio_latency_v4"]:
            data = int(line_data["ratio_latency_v4"].rstrip("%"))
            if data >= 100:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 7, line_data["ratio_latency_v4"], self.style_red)
            else:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 7, line_data["ratio_latency_v4"], self.style)
        else:
            self.worksheet_v4.write(GLOBAL_LINE_v4, 7, line_data["ratio_latency_v4"], self.style)

        if line_data["ratio_qps_v4"]:
            data = int(line_data["ratio_qps_v4"].rstrip("%"))
            if data >= 100:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 8, line_data["ratio_qps_v4"], self.style_red)
            else:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 8, line_data["ratio_qps_v4"], self.style)
        else:
            self.worksheet_v4.write(GLOBAL_LINE_v4, 8, line_data["ratio_qps_v4"], self.style) 

        if line_data["ratio_latency_v4_2"]:
            data = int(line_data["ratio_latency_v4_2"].rstrip("%"))
            if data >= 100:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 9, line_data["ratio_latency_v4_2"], self.style_red)
            else:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 9, line_data["ratio_latency_v4_2"], self.style)
        else:
            self.worksheet_v4.write(GLOBAL_LINE_v4, 9, line_data["ratio_latency_v4_2"], self.style)

        if line_data["ratio_qps_v4_2"]:
            data = int(line_data["ratio_qps_v4_2"].rstrip("%"))
            if data >= 100:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 10, line_data["ratio_qps_v4_2"], self.style_red)
            else:
                self.worksheet_v4.write(GLOBAL_LINE_v4, 10, line_data["ratio_qps_v4_2"], self.style)
        else:
            self.worksheet_v4.write(GLOBAL_LINE_v4, 10, line_data["ratio_qps_v4_2"], self.style)

#        if line_data["ratio_accuracy_rate_v4"]:
#            data = int(line_data["ratio_accuracy_rate_v4"].rstrip("%"))
#            if data == 100:
#                self.worksheet_v4.write(GLOBAL_LINE_v4, 10, line_data["ratio_accuracy_rate_v4"], self.style)
#            else:
#                self.worksheet_v4.write(GLOBAL_LINE_v4, 10, line_data["ratio_accuracy_rate_v4"], self.style_red)
#        else:
#            self.worksheet_v4.write(GLOBAL_LINE_v4, 10, line_data["ratio_accuracy_rate_v4"], self.style)

        GLOBAL_LINE_v4 += 1
        self.worksheet_v4.write(GLOBAL_LINE_v4, 2, line_data["Library_paddle"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 3, line_data["paddle_latency_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 4, line_data["paddle_qps_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 5, line_data["paddle_ratio_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 6, line_data["paddle_memory_v4"], self.style)

        GLOBAL_LINE_v4 += 1
        self.worksheet_v4.write(GLOBAL_LINE_v4, 2, line_data["Library_lego"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 3, line_data["lego_latency_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 4, line_data["lego_qps_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 5, line_data["lego_ratio_v4"], self.style)
        self.worksheet_v4.write(GLOBAL_LINE_v4, 6, line_data["lego_memory_v4"], self.style)


if __name__ == '__main__':
    # init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    global GLOBAL_LINE_5117
    global GLOBAL_LINE_v3
    global GLOBAL_LINE_v4

    # init excel
    wow = LoadExcel()
    # init config_parser
    try:
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/load_config.conf")
        threadsize_list=[1,2,6,10,12]
        #threadsize_list=[1]
        if len(sys.argv) == 2:
            #TODO
            model_list = sys.argv[1].split(",")
            for model in model_list:
                conf_name = "conf_%s" % model
                for thread_size in threadsize_list:
                    print "==========model: %s,thread_size: %s================" % (model, thread_size)
                    try:
                        db_name = cf.get(conf_name, "test_db") % thread_size
                    except Exception as e:
                        print ("[error]: Pls Check The Modle:%s input wrong!" % model)
                        sys.exit(1)
                    print db_name
                    trigger = LoadPerformance(db_name, model, thread_size)
                    line_data_5117, line_data_v3, line_data_v4 = trigger.make_excel_result()
                        
                    wow.put_in_excel_5117(line_data_5117)
                    wow.put_in_excel_v3(line_data_v3)
                    wow.put_in_excel_v4(line_data_v4)

                    GLOBAL_LINE_5117 += 1
                    GLOBAL_LINE_v3 += 1
                    GLOBAL_LINE_v4 += 1
                GLOBAL_LINE_5117 += 1
                GLOBAL_LINE_v3 += 1
                GLOBAL_LINE_v4 += 1
        else:
            print "[error]: usage:\"python fuck.py cnn_seg,yolo_lane_v2,.....\""
    except Exception as exception:
        print exception
        sys.exit(1)

    # save the excel
    del wow
