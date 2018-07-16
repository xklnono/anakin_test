#!/bin/bash

for file in `ls | grep -E model_batchsize_test_.*_p4 | grep -v perf`;
do
    if [ "$file" == "model_batchsize_test_segnet_p4" ];
    then
        echo "-------continue: $file"
    elif [ "$file" == "model_batchsize_test_diepsie_p4" ];
    then
        echo "-------continue: $file"
    else
        #./$file 1
        echo "$file 1"
        if [[ $? -ne 0 ]];then
            echo "[error]: ####fail: $file####"
            exit 1
#        else
#            echo "[info]: ####success: $file####"
        fi
    fi
done
