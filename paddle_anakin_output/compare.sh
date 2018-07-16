#!/bin/bash
export PATH=/home/qa_work/anaconda2/bin/:$PATH
###chinese_ner lexical_analysis sequence_labeling ###

CUR_PATH=`pwd`
chinese_ner_model="/home/qa_work/CI/workspace/sys_anakin_compare_output/chinese_ner/models"
chinese_ner_data="/home/qa_work/CI/workspace/sys_anakin_compare_output/chinese_ner/input_file"

sequence_labeling_model="/home/qa_work/CI/workspace/sys_anakin_compare_output/sequence_labeling/models"
sequence_labeling_data="/home/qa_work/CI/workspace/sys_anakin_compare_output/sequence_labeling/input_file"

lexical_analysis_model="/home/qa_work/CI/workspace/sys_anakin_compare_output/lexical_analysis/models"
lexical_analysis_data="/home/qa_work/CI/workspace/sys_anakin_compare_output/lexical_analysis/input_file"

sequence_labeling_UT="/home/qa_work/CI/workspace/paddle_anakin_output/fluid_sequence_labeling"
chinese_ner_UT="/home/qa_work/CI/workspace/paddle_anakin_output/fluid_chinese_ner"

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

anakin_UT="/home/qa_work/CI/workspace/sys_anakin_merge_build/anakin2/"
compare_path="/home/qa_work/xukailu/module_check/check_compare/"

##########langugae model##############
##(1)run the language model use paddle
if [ -d $lang_paddle_output_path ]; then
    cd $lang_paddle_output_path
    rm -rf *
fi

cd $CUR_PATH/fluid_lang_text
python language_many_output.py $lang_paddle_model_path $lang_input_path $lang_paddle_output_path $lang_output_name
##(2)run the languae model use anakin
if [ -d $anakin_output ];then
    cd $anakin_output
    rm -rf *
fi

cd $anakin_UT
./output/unit_test/net_exec_x86_oneinput $lang_anakin_model_path $lang_input_path 1 1 ${lang_output_name}_out
##(3) compare output
cd $CUR_PATH
tmp= `python compare_result.py $lang_paddle_output_path $anakin_output`
if [ -z $tmp ];then
    echo "PASS" > $CUR_PATH/result.txt
else
    echo "NO_PASS" > $CUR_PATH/result.txt
fi

##########text_classification model##############
##(1)run the text_classification model use paddle
if [ -d $text_paddle_output_path ]; then
    cd $text_paddle_output_path
    rm -rf *
fi
cd $CUR_PATH/fluid_lang_text
python language_many_output.py $text_paddle_model_path $text_input_path $text_paddle_output_path $text_output_name
##(2)run the chinese model use anakin
if [ -d $anakin_output ];then
    cd $anakin_output
    rm -rf *
fi

cd $anakin_UT
./output/unit_test/net_exec_x86_oneinput $text_anakin_model_path $text_input_path 1 1 ${text_output_name}_gout
##(3) compare output
cd $CUR_PATH
tmp= `python compare_result.py $text_paddle_output_path $anakin_output`
if [ -z $tmp ];then
    echo "PASS" >>$CUR_PATH/result.txt
else
    echo "NO_PASS" >> $CUR_PATH/result.txt
fi


############sequence_labeling model#############
##(1)run the sequence_labeling model use paddle
cd $sequence_labeling_UT
python infer.py sequence_labeling > $CUR_PATH/output/sequence_labeling_paddle.txt
##(2)run the sequence_labeling model use anakin
cd $anakin_UT
./output/unit_test/chinese_ner_test $sequence_labeling_model $sequence_labeling_data/perf-eval.legoraw  > $CUR_PATH/output/sequence_labeling_anakin.txt
##(3)compare the output
cd $compare_path
result=`python beyond_compared.py $CUR_PATH/output/sequence_labeling_paddle.txt $CUR_PATH/output/sequence_labeling_anakin.txt yolo_lane_v2`
echo $result 
echo $result >> $CUR_PATH/result.txt

########### lexical_analysis model ###########
##(1)run the sequence_labeling model use paddle
cd $sequence_labeling_UT
python infer.py lexical_analysis > $CUR_PATH/output/lexical_analysis_paddle.txt
##(2)run the sequence_labeling model use anakin
cd $anakin_UT
./output/unit_test/chinese_ner_test $lexical_analysis_model $lexical_analysis_data/perf-eval.legoraw  > $CUR_PATH/output/lexical_analysis_anakin.txt
##(3)compare the output
cd $compare_path
result=`python beyond_compared.py $CUR_PATH/output/lexical_analysis_paddle.txt $CUR_PATH/output/lexical_analysis_anakin.txt yolo_lane_v2`
echo $result
echo $result >> $CUR_PATH/result.txt

########### chinese_ner model ###############
#sed -i '/#define WITH_MENTION/s/^/\/\//g' $anakin_UT/test/framework/net/chinese_ner_test.cpp  //add //
sed -i '/define WITH_MENTION/s/^\/\///g' $anakin_UT/test/framework/net/chinese_ner_test.cpp    
cd $anakin_UT/build
make -j24

##(1)run the sequence_labeling model use paddle
cd $chinese_ner_UT
python infer.py  > $CUR_PATH/output/chinese_ner_paddle.txt
##(2)run the sequence_labeling model use anakin
cd $anakin_UT
./output/unit_test/chinese_ner_test $chinese_ner_model $chinese_ner_data/data_file  > $CUR_PATH/output/chinese_ner_anakin.txt
##(3)compare the output
cd $compare_path
result=`python beyond_compared.py $CUR_PATH/output/chinese_ner_paddle.txt $CUR_PATH/output/chinese_ner_anakin.txt yolo_lane_v2`
echo $result
echo $result >> $CUR_PATH/result.txt

