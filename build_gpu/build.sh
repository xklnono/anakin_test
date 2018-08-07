#!/bin/bash

ANAKIN2_UT_PATH="/home/qa_work/CI/workspace/sys_anakin_merge_build/output"
TENSORRT_UT_PATH="/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output"
TENSORRT_K1200_UT_PATH="/home/qa_work/CI/workspace/sys_tensorRT_merge_build/output_k1200"
WORK_PATH=`pwd`

declare -a model_name
declare -a model_result_dir

#model_name=(attribute blur feature_patch0 feature_patch1 feature_patch2 feature_patch3 feature_patch4 feature_patch5 feature_patch6 remark_demark remark_super score alignment_stage1 alignment_stage2 occlusion inception mobilenet Resnet101 Resnet50 ssd liveness vgg16 cnn_seg yolo_lane_v2 yolo_camera_detector densenbox diepsie mobilenet_ssd_fluid mobilenet_ssd_caffe se_resnext50_caffe se_resnext50_fluid vgg19 segnet mobilenet_v2 yolo mainbody mobilenetssd  animal_v2 plant_v2 new256 se-ResNeXt50 car_ssd mapdemo)

model_name=('vgg16' 'Resnet50' 'Resnet101' 'mobilenet_v2' 'yolo' 'mobilenet')

#vis_model=(mainbody mobilenetssd ocr20)

model_result_dir=(images_output output models RT_models time multi_thread_time paddle_output caffe_output input_file)

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
##(2)clear the output dir
for modlename in ${model_name[*]}
do
    if [ -d "/home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/images_output" ] ;then
        cd /home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/images_output
        rm -rf *
    fi
    if [ -d "/home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/output" ] ;then
        cd /home/qa_work/CI/workspace/sys_anakin_compare_output/${modlename}/output
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
./build_anakin.sh anakin_p4
./build_anakin.sh anakin_k1200
##(4)compile the tensorRT UT
exit
cd $WORK_PATH
./build_tensorRT.sh tensorRT p4
./build_tensorRT.sh tensorRT k1200
