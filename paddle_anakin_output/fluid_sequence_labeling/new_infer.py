from __future__ import division

import numpy as np
import reader
import paddle.fluid as fluid
import paddle.v2 as paddle
import contextlib
import unittest
import pdb
from  datetime import datetime

def to_lodtensor(data, place):
    """
    convert data to lodtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def infer(model_path, batch_size, test_data_dir):
 
    use_cuda = False   
    test_data = paddle.batch(
        reader.file_reader(test_data_dir),
        batch_size=batch_size)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    t_total = 0
    count = 0
    elapse = list()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        for data in test_data():
            count += 1
            word = to_lodtensor(map(lambda x: x[0], data), place)
            t1 = datetime.now()
            crf_decode = exe.run(inference_program,
                                 feed={"word":word},
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
            t2 = datetime.now()
            t_total = ((t2 - t1).total_seconds())
            elapse.append(t_total)
          
    avg_time = 0.0
    for t in elapse:
        avg_time += t
    print avg_time / len(elapse)

if __name__ == "__main__":
#    t = 0
#    for i in range(100):
    infer(
            model_path="/home/zhangshuai20/sequence_labeling/model/params_batch_450000",
            batch_size=60,
            test_data_dir="/home/zhangshuai20/sequence_labeling/test_set")
    #print "the avg. time is : %f ms" % t
 

