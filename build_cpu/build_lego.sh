#!/bin/bash

LEGO_PATH="/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/"
PAIRWISE_PATH=$LEGO_PATH"/pairwise_demo"

##(1)bcloud local to get lego's evn
cd $LEGO_PATH
bcloud local
sh build_mkl.sh

##(2)compile the UT
cd $PAIRWISE_PATH
sh build_mkl.sh

##(3)export the LD_PATH
export LD_LIBRARY_PATH=/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/mkl_lib/intel64:$LD_LIBRARY_PATH


