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

GLOBAL_LINE = 1

class LoadPerformance(object):
    """
    init
    """
    def __init__(self, db_name, model, batch_size):
        """
        init
        """
        # init mysql
        self.mysql = LoadCommon(db_name)
        self.batch_size = batch_size
        self.model = model

    def make_excel_result(self):
        """
        analysis the excel data
        1. Net_name 
        2. Batch_size
        3. Library: RT
        4. tensorRT Latency (ms)
        5. RT Memory (MB)
        6. Library: Anakin2
        7. anakin2 Latency (ms)
        8. RT Memory (MB)
        9. anakin/tensorrt latency
        10. anakin/tensorrt memory
        """
        # 1.Net_name
        net_name = model
        # 2. Batch_size
        batch_size = self.batch_size
        # 3. Library: RT
        Library_RT = "RT"
        # 6. Library: Anakin2
        Library_anakin2 = "Anakin2"

        # 4. tensorRT Latency (ms)
        tensorRT_latency_p4 = self.mysql.select_tensorRT_latency("p4")
        tensorRT_latency_k1200 = self.mysql.select_tensorRT_latency("k1200")
        print "tensorRT_latency_p4: %s" % tensorRT_latency_p4
        print "tensorRT_latency_k1200: %s" % tensorRT_latency_k1200
        # 5. RT Memory (MB)
        tensorRT_memory_p4  = self.mysql.select_tensorRT_memory("p4")
        tensorRT_memory_k1200  = self.mysql.select_tensorRT_memory("k1200")
        print "tensorRT_memory_p4: %s" % tensorRT_memory_p4
        print "tensorRT_memory_k1200: %s" % tensorRT_memory_k1200
        # 7. anakin2 Latency (ms)
        anakin2_latency_p4  = self.mysql.select_anakin2_latency("p4")
        anakin2_latency_k1200  = self.mysql.select_anakin2_latency("k1200")
        print "anakin2_latency_p4: %s" % anakin2_latency_p4
        print "anakin2_latency_k1200: %s" % anakin2_latency_k1200
        # 8. RT Memory (MB)
        anakin2_memory_p4   = self.mysql.select_anakin2_memory("p4")
        anakin2_memory_k1200   = self.mysql.select_anakin2_memory("k1200")
        print "anakin2_memory_p4: %s" % anakin2_memory_p4
        print "anakin2_memory_k1200: %s" % anakin2_memory_k1200

        # 9. anakin/tensorrt latency
        if tensorRT_latency_p4 and anakin2_latency_p4:
            #ratio_latency_p4 = string((float(anakin2_latency_p4) / float(tensorRT_latency_p4)) * 100)
            ratio_latency_p4 = str(int((float(anakin2_latency_p4) / float(tensorRT_latency_p4)) * 100)) + "%"
        else:
            ratio_latency_p4 = None
        if tensorRT_latency_k1200 and anakin2_latency_k1200:
            ratio_latency_k1200 = str(int((float(anakin2_latency_k1200) / float(tensorRT_latency_k1200)) * 100)) + "%"
        else:
            ratio_latency_k1200 = None
        print "ratio_latency_p4: %s" % ratio_latency_p4
        print "ratio_latency_k1200: %s" % ratio_latency_k1200

        # 10. anakin/tensorrt memory
        if tensorRT_memory_p4 and anakin2_memory_p4:
            ratio_memory_p4 = str(int((float(anakin2_memory_p4) / float(tensorRT_memory_p4)) * 100)) + "%"
        else:
            ratio_memory_p4 = None

        if tensorRT_memory_k1200 and anakin2_memory_k1200:
            ratio_memory_k1200 = str(int((float(anakin2_memory_k1200) / float(tensorRT_memory_k1200)) * 100)) + "%"
        else:
            ratio_memory_k1200 = None
        print "ratio_memory_p4: %s" % ratio_memory_p4
        print "ratio_memory_k1200: %s" % ratio_memory_k1200
      
        line_data_p4 = {}
        line_data_p4["net_name"] = net_name
        line_data_p4["batch_size"] = batch_size
        line_data_p4["Library_RT"] = Library_RT
        line_data_p4["tensorRT_latency_p4"] = tensorRT_latency_p4
        line_data_p4["tensorRT_memory_p4"] = tensorRT_memory_p4
        line_data_p4["Library_anakin2"] = Library_anakin2 
        line_data_p4["anakin2_latency_p4"] = anakin2_latency_p4
        line_data_p4["anakin2_memory_p4"] = anakin2_memory_p4
        line_data_p4["ratio_latency_p4"] = ratio_latency_p4
        line_data_p4["ratio_memory_p4"] = ratio_memory_p4

        line_data_k1200 = {}
        line_data_k1200["net_name"] = net_name
        line_data_k1200["batch_size"] = batch_size
        line_data_k1200["Library_RT"] = Library_RT
        line_data_k1200["tensorRT_latency_k1200"] = tensorRT_latency_k1200
        line_data_k1200["tensorRT_memory_k1200"] = tensorRT_memory_k1200
        line_data_k1200["Library_anakin2"] = Library_anakin2
        line_data_k1200["anakin2_latency_k1200"] = anakin2_latency_k1200
        line_data_k1200["anakin2_memory_k1200"] = anakin2_memory_k1200
        line_data_k1200["ratio_latency_k1200"] = ratio_latency_k1200
        line_data_k1200["ratio_memory_k1200"] = ratio_memory_k1200

        return line_data_p4, line_data_k1200

       
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
        self.worksheet_p4 = self.workbook.add_sheet("p4(anakin2 vs tensorRT)")
        self.worksheet_k1200 = self.workbook.add_sheet("k1200(anakin2 vs tensorRT)")

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
        col = self.worksheet_p4.col(0)
        col.width = 256 * 25
        col1 = self.worksheet_k1200.col(0)
        col1.width = 256 * 25
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
        self.worksheet_p4.write(0, 0, "Net_Name", self.style_background0)
        self.worksheet_k1200.write(0, 0, "Net_Name", self.style_background0)

        col = self.worksheet_p4.col(1)
        col.width = 256 * 15
        col1 = self.worksheet_k1200.col(1)
        col1.width = 256 * 15
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
        self.worksheet_p4.write(0, 1, "Batch_Size", self.style_background1)
        self.worksheet_k1200.write(0, 1, "Batch_Size", self.style_background1)

        col = self.worksheet_p4.col(2)
        col.width = 256 * 15
        col1 = self.worksheet_k1200.col(2)
        col1.width = 256 * 15
        self.style_background2 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background2.font = font
        self.style_background2.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 44
        self.style_background2.pattern = pattern
        self.worksheet_p4.write(0, 2, "Library", self.style_background2)
        self.worksheet_k1200.write(0, 2, "Library", self.style_background2)

        col = self.worksheet_p4.col(3)
        col.width = 256 * 20
        col1 = self.worksheet_k1200.col(3)
        col1.width = 256 * 20
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
        self.worksheet_p4.write(0, 3, "RT latency (ms)", self.style_background3)
        self.worksheet_k1200.write(0, 3, "RT latency (ms)", self.style_background3)

        col = self.worksheet_p4.col(4)
        col.width = 256 * 20
        col1 = self.worksheet_k1200.col(4)
        col1.width = 256 * 20
        self.style_background4 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background4.font = font
        self.style_background4.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 47
        self.style_background4.pattern = pattern
        self.worksheet_p4.write(0, 4, "RT Memory (MB)", self.style_background4)
        self.worksheet_k1200.write(0, 4, "RT Memory (MB)", self.style_background4)

        col = self.worksheet_p4.col(5)
        col.width = 256 * 15
        col1 = self.worksheet_k1200.col(5)
        col1.width = 256 * 15
        self.style_background5 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background5.font = font
        self.style_background5.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 44
        self.style_background5.pattern = pattern
        self.worksheet_p4.write(0, 5, "Library", self.style_background5)
        self.worksheet_k1200.write(0, 5, "Library", self.style_background5)

        col = self.worksheet_p4.col(6)
        col.width = 256 * 20
        col1 = self.worksheet_k1200.col(6)
        col1.width = 256 * 20
        self.style_background6 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background6.font = font
        self.style_background6.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 42
        self.style_background6.pattern = pattern
        self.worksheet_p4.write(0, 6, "anakin2 Latency (ms)", self.style_background6)
        self.worksheet_k1200.write(0, 6, "anakin2 Latency (ms)", self.style_background6)

        col = self.worksheet_p4.col(7)
        col.width = 256 * 20
        col1 = self.worksheet_k1200.col(7)
        col1.width = 256 * 20
        self.style_background7 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background7.font = font
        self.style_background7.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 47
        self.style_background7.pattern = pattern
        self.worksheet_p4.write(0, 7, "anakin2 Memory (MB)", self.style_background7)
        self.worksheet_k1200.write(0, 7, "anakin2 Memory (MB)", self.style_background7)

        col = self.worksheet_p4.col(8)
        col.width = 256 * 20
        col1 = self.worksheet_k1200.col(8)
        col1.width = 256 * 20
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
        self.worksheet_p4.write(0, 8, "anakin/tensorRT\nlatency", self.style_background8)
        self.worksheet_k1200.write(0, 8, "anakin/tensorRT\nlatency", self.style_background8)

        col = self.worksheet_p4.col(9)
        col.width = 256 * 20
        col1 = self.worksheet_k1200.col(9)
        col1.width = 256 * 20
        self.style_background9 = xlwt.XFStyle() # Create Style
        font = xlwt.Font()
        font.name = 'PT Mono'
        font.height = 240
        font.colour_index = 0
        self.style_background9.font = font
        self.style_background9.alignment = alignment
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = 53
        self.style_background9.pattern = pattern
        self.worksheet_p4.write(0, 9, "anakin/tensorrt\nmemory", self.style_background9)
        self.worksheet_k1200.write(0, 9, "anakin/tensorrt\nmemory", self.style_background9)

    def __del__(self):
        """
        del
        """
        # save the excel 
        self.workbook.save('anakin_VS_RT.xls')

    def put_in_excel_p4(self, line_data):
        """
        input excel
        """
        global GLOBAL_LINE
        # input excel
        # 1. p4 sheet
        self.worksheet_p4.write(GLOBAL_LINE, 0, line_data["net_name"], self.style)
        self.worksheet_p4.write(GLOBAL_LINE, 1, line_data["batch_size"], self.style)
        self.worksheet_p4.write(GLOBAL_LINE, 2, line_data["Library_RT"], self.style)
        self.worksheet_p4.write(GLOBAL_LINE, 3, line_data["tensorRT_latency_p4"], self.style)
        self.worksheet_p4.write(GLOBAL_LINE, 4, line_data["tensorRT_memory_p4"], self.style)
        self.worksheet_p4.write(GLOBAL_LINE, 5, line_data["Library_anakin2"], self.style)
        self.worksheet_p4.write(GLOBAL_LINE, 6, line_data["anakin2_latency_p4"], self.style)
        self.worksheet_p4.write(GLOBAL_LINE, 7, line_data["anakin2_memory_p4"], self.style)

        if line_data["ratio_latency_p4"]:
            data = int(line_data["ratio_latency_p4"].rstrip("%"))
            if data >= 100:
                self.worksheet_p4.write(GLOBAL_LINE, 8, line_data["ratio_latency_p4"], self.style_red)
            else:
                self.worksheet_p4.write(GLOBAL_LINE, 8, line_data["ratio_latency_p4"], self.style)
        else:
            self.worksheet_p4.write(GLOBAL_LINE, 8, line_data["ratio_latency_p4"], self.style)
     
        if line_data["ratio_memory_p4"]:
            data = int(line_data["ratio_memory_p4"].rstrip("%"))
            if data >= 100:
                self.worksheet_p4.write(GLOBAL_LINE, 9, line_data["ratio_memory_p4"], self.style_red)
            else:
                self.worksheet_p4.write(GLOBAL_LINE, 9, line_data["ratio_memory_p4"], self.style)
        else:
            self.worksheet_p4.write(GLOBAL_LINE, 9, line_data["ratio_memory_p4"], self.style)

    def put_in_excel_k1200(self, line_data):
        """
        input excel
        """
        global GLOBAL_LINE
        # input excel
        # 2. k1200 sheet
        self.worksheet_k1200.write(GLOBAL_LINE, 0, line_data["net_name"], self.style)
        self.worksheet_k1200.write(GLOBAL_LINE, 1, line_data["batch_size"], self.style)
        self.worksheet_k1200.write(GLOBAL_LINE, 2, line_data["Library_RT"], self.style)
        self.worksheet_k1200.write(GLOBAL_LINE, 3, line_data["tensorRT_latency_k1200"], self.style)
        self.worksheet_k1200.write(GLOBAL_LINE, 4, line_data["tensorRT_memory_k1200"], self.style)
        self.worksheet_k1200.write(GLOBAL_LINE, 5, line_data["Library_anakin2"], self.style)
        self.worksheet_k1200.write(GLOBAL_LINE, 6, line_data["anakin2_latency_k1200"], self.style)
        self.worksheet_k1200.write(GLOBAL_LINE, 7, line_data["anakin2_memory_k1200"], self.style)

        if line_data["ratio_latency_k1200"]:
            data = int(line_data["ratio_latency_k1200"].rstrip("%"))
            if data >= 100:
                self.worksheet_k1200.write(GLOBAL_LINE, 8, line_data["ratio_latency_k1200"], self.style_red)
            else:
                self.worksheet_k1200.write(GLOBAL_LINE, 8, line_data["ratio_latency_k1200"], self.style)
        else:
            self.worksheet_k1200.write(GLOBAL_LINE, 8, line_data["ratio_latency_k1200"], self.style)

        if line_data["ratio_memory_k1200"]:
            data = int(line_data["ratio_memory_k1200"].rstrip("%"))
            if data >= 100:
                self.worksheet_k1200.write(GLOBAL_LINE, 9, line_data["ratio_memory_k1200"], self.style_red)
            else:
                self.worksheet_k1200.write(GLOBAL_LINE, 9, line_data["ratio_memory_k1200"], self.style)
        else:
            self.worksheet_k1200.write(GLOBAL_LINE, 9, line_data["ratio_memory_k1200"], self.style) 

if __name__ == '__main__':
    # init mylogging
    logger = mylogging.init_log(logging.DEBUG)

    global GLOBAL_LINE

    # init excel
    wow = LoadExcel()
    # init config_parser
    try:
        cf = ConfigParser.ConfigParser()
        cf.read("../conf/load_config.conf")
        batchsize_list=[1,2,4,8,32]
        if len(sys.argv) == 2:
            #TODO
            model_list = sys.argv[1].split(",")
            for model in model_list:
                conf_name = "conf_%s" % model
                for batch_size in batchsize_list:
                    print "==========model: %s,batch_size: %s================" % (model, batch_size)
                    try:
                        db_name = cf.get(conf_name, "test_db") % batch_size
                    except Exception as e:
                        print ("[error]: Pls Check The Modle:%s input wrong!" % model)
                        sys.exit(1)
                    print db_name
                    trigger = LoadPerformance(db_name, model, batch_size)
                    line_data_p4, line_data_k1200 = trigger.make_excel_result()
                        
                    wow.put_in_excel_p4(line_data_p4)
                    wow.put_in_excel_k1200(line_data_k1200)

                    GLOBAL_LINE += 1
        else:
            print "[error]: usage:\"python fuck.py cnn_seg,yolo_lane_v2,.....\""
    except Exception as exception:
        print exception
        sys.exit(1)

    # save the excel
    del wow
