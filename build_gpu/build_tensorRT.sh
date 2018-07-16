#!/bin/bash
export LANG="zh_CN.UTF-8"

CUR_PATH=`pwd`
LPATH="/home/qa_work/CI/workspace/sys_tensorRT_merge_build"
UTF_PATH=$LPATH"/anakin1/adu-inference/test"
RUN_PATH=$LPATH"/anakin1/adu-inference"
OUTPUT=$LPATH"/anakin1/adu-inference/output/unit_test"
NEW_OUTPUT=$LPATH"/output"
K1200_OUTPUT=$LPATH"/output_k1200"

##(1)modify the UT.cpp
cd $CUR_PATH
python modify.py $1 $2

##(2)compile
cd $RUN_PATH
cd build
rm -rf *
cmake ..
make -j24


##(4)cp the output to the current dir
if [ $2 == "p4" ]; then
    cp $OUTPUT/rt_batchsize_test_* $NEW_OUTPUT
    cd $NEW_OUTPUT
    for i in `ls`;
    do
        cp $i ${i}_p4
    #cp $i ${i}_k1200
    done
elif [ $2 == "k1200" ]; then
    cp $OUTPUT/rt_batchsize_test_* $K1200_OUTPUT
    cd $K1200_OUTPUT
    for i in `ls`;
    do
        mv $i ${i}_k1200
    done
    cp $K1200_OUTPUT/rt_batchsize_test_* $NEW_OUTPUT
fi
