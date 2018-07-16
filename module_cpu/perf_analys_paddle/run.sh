#!/bin/bash

set -e


unset KMP_AFFINITY
export KMP_AFFINITY="granularity=fine,compact,0,0" # when HT if OFF
#export KMP_AFFINITY="granularity=fine,compact,1,0" # when HT is ON

# 1 socket for 8180
# echo 0 > /proc/sys/kernel/numa_balancing
core_num=`nproc`
threads_per_core=`lscpu | grep "per core" | awk -F ':' '{print $2}' | sed 's/^ *\| *$//g'`
sockets=`lscpu | grep "socket(s)" | awk -F ':' '{print $2}' | sed 's/^ *\| *$//g'`
let "core_num=$core_num / ( $threads_per_core * $sockets )"
#echo ${core_num}

core_idx=`expr ${core_num} - 1`
core_range='0-'${core_idx}

echo ${core_range}


#run_exec=/home/qa_work/CI/workspace/Paddle/build/paddle/fluid/inference/tests/book/

#taskset -c ${core_range} numactl -l  ${run_exec}test_inference_nlp -num_threads $1 
#taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 10 > $lang_res\_10.txt
#taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 6 > $lang_res\_6.txt 
#taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 2 > $lang_res\_2.txt 
#taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 1 > $lang_res\_1.txt 

