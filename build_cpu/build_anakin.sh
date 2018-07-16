#!/bin/bash
#export LANG="zh_CN.UTF-8"

CUR_PATH=`pwd`
COMPILE_ROOT_PATH="/home/qa_work/CI/workspace/sys_anakin_merge_build"
ANAKIN2_ROOT_PATH=$COMPILE_ROOT_PATH"/anakin2"
BUILD_SH_PATH=$ANAKIN2_ROOT_PATH"/tools"
RUN_YOLO_PATH=$ANAKIN2_ROOT_PATH"/output/unit_test"
NEW_FILE_PATH=$CUR_PATH"/new_file"
NEW_OUTPUT=$COMPILE_ROOT_PATH"/output"
DST_PATH="/home/qa_work/CI/workspace/sys_anakin_merge_build/output"

##(1)git clone code: anakin
#cd $COMPILE_ROOT_PATH
#if [ -e anakin2 ];
#then
#    rm -rf anakin2
#fi
#
#git clone https://github.com/PaddlePaddle/Anakin.git
#mv Anakin anakin2
#cd $ANAKIN2_ROOT_PATH
#git checkout developing
cd $ANAKIN2_ROOT_PATH

##(2)modify the CMakeLists.txt and find_modules.cmake to use the opencv
cp $ANAKIN2_ROOT_PATH/CMakeLists.txt $NEW_FILE_PATH/CMakeLists.txt
sed -i "/Select the build mode for GPU place/s/YES/NO/g" $NEW_FILE_PATH/CMakeLists.txt
sed -i "/Select the build mode for X86 place/s/NO/YES/g" $NEW_FILE_PATH/CMakeLists.txt
sed -i "/Use static opencv libs/s/YES/NO/g" $NEW_FILE_PATH/CMakeLists.txt

cp -f $NEW_FILE_PATH"/CMakeLists.txt" $ANAKIN2_ROOT_PATH

##(4)compile the code
if [ -d $ANAKIN2_ROOT_PATH ];then
    cd $ANAKIN2_ROOT_PATH
    rm -rf build
    rm -rf gpu_build
    rm -rf output
fi

cp -f $NEW_FILE_PATH/*_qa* $ANAKIN2_ROOT_PATH/test/framework/net/
mkdir build
cd build
cmake .. -DWITH_GPU=OFF -DWITH_FLUID_ONLY=ON -DWITH_TESTING=ON
cp $ANAKIN2_ROOT_PATH/build/CMakeFiles/extern_mklml.dir/build.make $NEW_FILE_PATH/build.make
sed -i "/http/s/paddlepaddledeps.cdn.bcebos.com/10.92.103.11:8080/g" $NEW_FILE_PATH/build.make
cp -f $NEW_FILE_PATH"/build.make" $ANAKIN2_ROOT_PATH/build/CMakeFiles/extern_mklml.dir/build.make
make -j24
##(5)cp UT to the output dir

cpu_name=`cat /proc/cpuinfo |grep name |head -n 1 | awk '{print $7}'`
if [ $cpu_name == "5117M" ]; then
    cpu_name=${cpu_name%?}  
    echo ${cpu_name}  
else
    cpu_name=`cat /proc/cpuinfo |grep name |head -n 1 | awk '{print $8}'`
fi

cp $RUN_YOLO_PATH/*qa* $NEW_OUTPUT
cd $NEW_OUTPUT
for i in `ls`;
do  
    if [ `echo $i |awk -F'_' '{print $NF}'` != ${cpu_name} ];then
        mv $i ${i%_*}_${cpu_name}
    fi
done
