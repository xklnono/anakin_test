#!/bin/bash

CUR_PATH=`pwd`
lang_paddle_model_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/language/fluid_models'
lang_input_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/language/input_file/fake_realtitle.test'
lang_paddle_output_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/language/paddle_output/'
lang_output_name='fc_1.tmp_2'
lang_anakin_model_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/language/models'

text_paddle_model_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/text_classification/fluid_models'
text_input_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/text_classification/input_file/out.ids.txt'
text_paddle_output_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/text_classification/paddle_output/'
text_output_name='fc_3.tmp_2'
text_anakin_model_path='/home/qa_work/CI/workspace/sys_anakin_compare_output/text_classification/models'

anakin_output='/home/qa_work/CI/workspace/sys_anakin_compare_output/anakin_output/'
anakin_ut_path='/home/qa_work/CI/workspace/sys_anakin_merge_build/anakin2/output/unit_test'

##########langugae model##############
##(1)run the language model use paddle
#cd $CUR_PATH
#python language_many_output.py $lang_paddle_model_path $lang_input_path $lang_paddle_output_path $lang_output_name
###(2)run the languae model use anakin
#if [ -d $anakin_output ];then
#    cd $anakin_output
#    rm -rf *
#fi
#
#cd $anakin_ut_path
#./net_exec_x86_oneinput $lang_anakin_model_path $lang_input_path 1 1 ${lang_output_name}_out
###(3) compare output
#cd $CUR_PATH
#tmp= `python compare_result.py £$lang_paddle_output_path $anakin_output`
#if [ -z $tmp ];then
#    echo "PASS"
#else
#    echo "NO_PASS"
#fi
#
#exit

##########chinese_ner model##############
##(1)run the chinese_ner model use paddle
#cd $CUR_PATH
#python language_many_output.py $text_paddle_model_path $text_input_path $text_paddle_output_path $text_output_name
##(2)run the chinese model use anakin
#if [ -d $anakin_output ];then
#    cd $anakin_output
#    rm -rf *
#fi

cd $anakin_ut_path
./net_exec_x86_oneinput $text_anakin_model_path $text_input_path 1 1 ${text_output_name}_gout
##(3) compare output
cd $CUR_PATH
tmp= `python compare_result.py $text_paddle_output_path $anakin_output`
if [ -z $tmp ];then
    echo "PASS"
else
    echo "NO_PASS"
fi
