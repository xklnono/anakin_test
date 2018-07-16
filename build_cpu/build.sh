#!/bin/bash

ANAKIN2_UT_PATH="/home/qa_work/CI/workspace/sys_anakin_merge_build/output"
TENSORRT_UT_PATH="/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output"
TENSORRT_K1200_UT_PATH="/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output_k1200"
WORK_PATH=`pwd`

declare -a model_name
declare -a model_result_dir

model_name=(chinese_ner language_model neural_machine_translation sequence_labeling text_classification)

model_result_dir=(images_output output models fluid_models time multi_thread_time paddle_output input_file)

##(1)mkdir for new model
for modlename in ${model_name[*]}
do
    if [ ! -d "/home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/" ]; then
        cd /home/qa_work/CI/workspace/sys_anakin_compare_output
        mkdir ${modlename}
        for dir in ${model_result_dir[*]}
        do 
            mkdir ${modlename}/${dir}
        done
    else
       echo "has no new models"
    fi
done
exit
##(2)clear the output dir
for modlename in ${model_name[*]}
do
    if [ -d "/home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/images_output" ] ;then
        cd /home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/images_output
        rm -rf *
    fi
    if [ -d "/home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/output" ] ;then
        cd /home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/paddle_output
        rm -rf *
    fi
done

##(2)clear the UT dir
if [ -d $ANAKIN2_UT_PATH ]; then
    cd $ANAKIN2_UT_PATH
    echo `pwd`
    rm -rf *
fi

#if [ -d $TENSORRT_UT_PATH ];then
#    cd $TENSORRT_UT_PATH
#    echo `pwd`
#    rm -rf *
#fi
#
#if [ -d $TENSORRT_K1200_UT_PATH ];then
#    cd $TENSORRT_K1200_UT_PATH
#    echo `pwd`
#    rm -rf *
#fi
##(3)compile the anakin UT
cd $WORK_PATH
./build_anakin.sh
exit
##(4)compile the paddle UT
cd $WORK_PATH
./build_paddle.sh

cd $WORK_PATH
./build_lego.sh
