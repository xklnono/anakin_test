#!/bin/bash
#export LANG="zh_CN.UTF-8"

CUR_PATH=`pwd`
PADDLE_ROOT="/home/qa_work/CI/workspace/Paddle/"
BUILD_PATH="/home/qa_work/CI/workspace/Paddle/build"
UT_PATH="/home/qa_work/CI/workspace/Paddle/build/paddle/fluid/inference/tests/book"
OUTPUT_PATH="/home/qa_work/CI/workspace/Paddle/output"

declare -a model_name
#model_name=(chinese_ner  language  sequence_labeling  text_classification)
model_name=(sequence_labeling )
##(1)get cpuinfo
cpu_name=`cat /proc/cpuinfo |grep name |head -n 1 | awk '{print $7}'`
if [ $cpu_name == "5117M" ]; then
    cpu_name=${cpu_name%?}
    echo $cpu_name
else
    cpu_name=`cat /proc/cpuinfo |grep name |head -n 1 | awk '{print $8}'`
fi
for item in ${model_name[@]}
do
    cd $CUR_PATH
    python modify_paddle.py $item
    cd $BUILD_PATH
    cmake .. -DWITH_GPU=OFF -DWITH_FLUID_ONLY=ON -DWITH_TESTING=ON
    make -j24
    cp $BUILD_PATH/paddle/fluid/inference/tests/book/test_inference_nlp $OUTPUT_PATH/test_inference_nlp_${item}_${cpu_name}
done


