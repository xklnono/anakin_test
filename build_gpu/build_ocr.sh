#!/bin/bash

ANAKIN2_UT_PATH="/home/qa_work/CI/workspace/sys_anakin_merge_build/output"
TENSORRT_UT_PATH="/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output"
TENSORRT_K1200_UT_PATH="/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output_k1200"
WORK_PATH=`pwd`

cd $WORK_PATH
./build_anakin_ocr.sh anakin_p4
./build_anakin_ocr.sh anakin_k1200
