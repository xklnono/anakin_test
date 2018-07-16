#!/bin/bash
#export LANG="zh_CN.UTF-8"

CUR_PATH=`pwd`
COMPILE_ROOT_PATH=$2
ANAKIN2_ROOT_PATH=$COMPILE_ROOT_PATH
BUILD_SH_PATH=$ANAKIN2_ROOT_PATH"/tools"
RUN_YOLO_PATH=$ANAKIN2_ROOT_PATH"/output/unit_test"
NEW_FILE_PATH=$CUR_PATH"/new_file"
DST_PATH=$CUR_PATH"/output"

##(1)git clone code: anakin
#cd $COMPILE_ROOT_PATH
#if [ -e output ];then
#    rm -rf output
#fi
#mkdir -p output
#if [ -e anakin2 ];
#then
#    rm -rf anakin2
#fi
#
#git clone https://github.com/PaddlePaddle/Anakin.git
#mv Anakin anakin2
#
#cd $ANAKIN2_ROOT_PATH
#git checkout developing
cd $ANAKIN2_ROOT_PATH

##(2)modify the CMakeLists.txt and find_modules.cmake to use the opencv
cp $ANAKIN2_ROOT_PATH/CMakeLists.txt $NEW_FILE_PATH/CMakeLists.txt
sed -i "/USE_OPENCV/s/NO/YES/g" $NEW_FILE_PATH/CMakeLists.txt
sed -i "/Select the build mode for X86 place/s/YES/NO/g" $NEW_FILE_PATH/CMakeLists.txt
sed -i "/Select the build mode for ARM place/s/YES/NO/g" $NEW_FILE_PATH/CMakeLists.txt
sed -i '/USE_OPENCV/s/^#//' $NEW_FILE_PATH/CMakeLists.txt 
sed -i "/ENABLE_OP_TIMER/s/YES/NO/g" $NEW_FILE_PATH/CMakeLists.txt
sed -i "/Build anakin lite components/s/NOT USE_ARM_PLACE/USE_GPU_PLACE/g" $NEW_FILE_PATH/CMakeLists.txt
num=`sed -n '/Protobuf REQUIRED/=' $ANAKIN2_ROOT_PATH/cmake/find_modules.cmake`
let "num=$num+1"
sed -i "${num}i\set(PROTOBUF_INCLUDE_DIR \"/home/qa_work/.jumbo/include\")"  $ANAKIN2_ROOT_PATH/cmake/find_modules.cmake 
echo $num
if [ $1 == "anakin_k1200" ]; then
    sed -i "/SELECTED_SASS_TARGET_ARCH/s/61/50/g" $NEW_FILE_PATH/CMakeLists.txt
fi
sed -i "/Enable DEBUG(default) mode/s/YES/NO/g" $NEW_FILE_PATH/CMakeLists.txt

cp -f $NEW_FILE_PATH"/CMakeLists.txt" $ANAKIN2_ROOT_PATH
cp -f $NEW_FILE_PATH"/find_modules.cmake" $ANAKIN2_ROOT_PATH"/cmake/find_modules.cmake"
cp -f $NEW_FILE_PATH"/net_test.h" $ANAKIN2_ROOT_PATH"/test/framework/net"
#modify cudnn path in cmake/cuda.cmake
sed -i "/CUDNN_INCLUDE_DIR cudnn.h PATHS/s/include//g" $ANAKIN2_ROOT_PATH/cmake/cuda.cmake
##(3)modify the specify context for every modle
cd $CUR_PATH
echo $1
echo $2
python modify.py $1 $2 
python modify_vis.py $1 $2 
##(4)compile the code
if [ -d $ANAKIN2_ROOT_PATH ];then
    cd $ANAKIN2_ROOT_PATH
    rm -rf build
    rm -rf gpu_build
    rm -rf output
fi
mkdir build
cd build
cmake ..
make -j24

##(5)cp UT to the output dir
#cp $RUN_YOLO_PATH/model_batchsize_test_* $DST_PATH
    

