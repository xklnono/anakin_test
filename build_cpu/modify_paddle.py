#!/bin/python
import os
import sys
import shutil
import re
import fileinput

def modify_paddle_file(modelname):
    cur_path = os.getcwd()
    src_path = os.path.join(cur_path, "new_file")
    src_name = "test_inference_nlp.cc." + modelname
    print src_name
    src_abs = os.path.join(src_path, src_name)
    tmp_file_name = "tmp.cpp"
    tmp_abs = os.path.join(src_path, tmp_file_name)
    dst_path = "/home/qa_work/CI/workspace/Paddle/paddle/fluid/inference/tests/book/"
    #dst_path = os.path.join(work_path, "anakin2/test/framework/net")
    dst_name = os.path.join(dst_path, src_name.split('.')[0] + '.' + src_name.split('.')[1])
    print dst_name
    shutil.copyfile(src_abs, tmp_abs)
    shutil.move(tmp_abs, dst_name)
if len(sys.argv) == 2:
    modify_paddle_file(sys.argv[1])
