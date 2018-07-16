from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import os
import sys
from paddle.fluid import debugger
from paddle.fluid.framework import Program, OpProtoHolder
import paddle.fluid.core as core
from paddle.fluid.executor import Executor

class FluidModelTest:

    def __init__(self, model_path, is_get_every_tensor=True,gpu_id=-1):
        if gpu_id >= 0:
            self.place = fluid.CUDAPlace(gpu_id)
        else:
            self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.scope = fluid.core.Scope()

        with fluid.scope_guard(self.scope):
            if os.path.exists(ModelPath + 'model') and os.path.exists(ModelPath + 'params'):
                [self.net_program, self.model_old_feed_file, self.model_old_fetch_file] = \
                    fluid.io.load_inference_model(ModelPath, self.exe, 'model', 'params')
            else:
                [self.net_program, self.model_old_feed_file, self.model_old_fetch_file] = \
                    fluid.io.load_inference_model(ModelPath, self.exe)
            temp_list=[]
            for fetch_poto in  self.model_old_fetch_file:
                temp_list.append(fetch_poto.name)
            self.model_old_fetch_file=temp_list

            self.global_block = self.net_program.global_block()
            self.model_old_fetch = []
            self.model_old_feed = []
            for old_op in list(self.global_block.ops):
                if old_op.type == 'fetch':
                    self.model_old_fetch.append(old_op.desc.input_arg_names()[0])
                elif old_op.type == 'feed':
                    self.model_old_feed.append(old_op.desc.output_arg_names()[0])

            self.fetch_list = self.__fetch_tmp_vars(self.global_block, self.model_old_fetch,is_get_every_tensor)

    def __fetch_tmp_vars(self, block, old_fetch_list,is_get_every_tensor=True):
        fetch_var = block.var('fetch')
        new_fetch_vars = []
        for var_name in old_fetch_list:
            var = block.var(var_name)
            new_fetch_vars.append(var)
        if not is_get_every_tensor:
            return new_fetch_vars
        i = len(old_fetch_list)
        var_names_list = block.vars.keys()
        for var_name in var_names_list:
            if '.tmp_' in var_name and var_name not in old_fetch_list:
                var = block.var(var_name)
                new_fetch_vars.append(var)
                block.append_op(
                    type='fetch',
                    inputs={'X': [var_name]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})
                i = i + 1
        return new_fetch_vars

    def __check_fetch_feed_equal(self):
        if set(self.model_old_fetch) != set(self.model_old_fetch_file):
            print('[warning!!] fetch file not equal')
        if set(self.model_old_feed) != set(self.model_old_feed_file):
            print('[warning!!] feed file not equal')

    def run(self, feed_dic):

        self.__check_fetch_feed_equal()

        with fluid.scope_guard(self.scope):
            inner_dic = {}
            if type(feed_dic) == type([]):
                if type(feed_dic[0])==type([]):
                    inner_dic[self.model_old_feed[0]]=self.to_lodtensor(feed_dic)
                    print('ok')
                else:
                    assert len(feed_dic) == len(self.model_old_feed), \
                        'input size must match {}!={}'.format(len(feed_dic), len(self.model_old_feed))
                    for i, feed_name in enumerate(self.model_old_feed):
                        inner_dic[feed_name] = feed_dic[i]
            elif type(feed_dic) == type({}):
                inner_dic = feed_dic
            elif type(feed_dic)==type(np.array([])):
                assert len(self.model_old_feed)==1,'one numpy input is allowed'
                inner_dic[self.model_old_feed[0]]=feed_dic
            else:
                raise Exception('only support list and dict as input')

            print('fetch list',self.fetch_list)
            results = self.exe.run(program=self.net_program, feed=inner_dic,
                                   fetch_list=self.fetch_list, return_numpy=False)
            self.result={}
            for i,var in enumerate(self.fetch_list):
                self.result[var.name]=results[i]
            return self.result

    def save_diagram(self, localtion='./model_diagram.dot'):
        debugger.draw_block_graphviz(self.global_block, path=localtion)

    def save_np_results(self,location='./'):
        assert self.result!=None,'must save after run'
        for var_name in self.result.keys():
            np.save(location+var_name,np.array(self.result[var_name]))

    def show(self):
        info_dic = {}
        info_dic['model_old_fetch'] = self.model_old_fetch
        info_dic['model_old_feed'] = self.model_old_feed
        modle_info = {}
        modle_info['model_old_fetch_file'] = self.model_old_fetch_file
        modle_info['model_old_feed_file'] = self.model_old_feed_file
        print(info_dic)
        print(modle_info)
        self.__check_fetch_feed_equal()

    def to_lodtensor(self,data):
        """ convert to LODtensor """
        seq_lens = [len(seq) for seq in data]
        cur_len = 0
        lod = [cur_len]
        for l in seq_lens:
            cur_len += l
            lod.append(cur_len)
        flattened_data = np.concatenate(data, axis=0).astype("int64")
        flattened_data = flattened_data.reshape([len(flattened_data), 1])
        res = fluid.LoDTensor()
        res.set(flattened_data, self.place)
        res.set_lod([lod])
        return res


class FluidOpTest():
    def __init__(self,op_name,is_show=True):
        self.place = fluid.CPUPlace()
        self.op_name=op_name
        self.op_proto=OpProtoHolder.instance().get_op_proto(self.op_name)
        if is_show:
            self.show()

    def __append_input_output(self,block, op_proto, np_list, is_input):
        '''Insert VarDesc and generate Python variable instance'''
        proto_list = op_proto.inputs if is_input else op_proto.outputs

        def create_var(block, name, np_list, var_proto):
            if name not in np_list:
                assert var_proto.intermediate, "{} not found".format(name)
                shape = None
                lod_level = None
            else:
                np_value = np_list[name]
                if isinstance(np_value, tuple):
                    shape = list(np_value[0].shape)
                    lod_level = len(np_value[1])
                else:
                    shape = list(np_value.shape)
                    lod_level = 0
            return block.create_var(
                dtype="float32", shape=shape, lod_level=lod_level, name=name)

        var_dict = {}
        for var_proto in proto_list:
            var_name = str(var_proto.name)
            if is_input:
                if (var_name not in np_list) and var_proto.dispensable:
                    continue
                assert (var_name in np_list) or (var_proto.dispensable), \
                    "Missing {} as input".format(var_name)
            if var_proto.duplicable:
                assert isinstance(np_list[var_name], list), \
                    "Duplicable {} should be set as list".format(var_name)
                var_list = []
                for (name, np_value) in np_list[var_name]:
                    var_list.append(
                        create_var(block, name, {name: np_value}, var_proto))
                var_dict[var_name] = var_list
            else:
                var_dict[var_name] = create_var(block, var_name, np_list, var_proto)

        return var_dict


    def show(self):
        print(self.op_proto)

    def __feed_var(self,input_vars, real_inputs):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in real_inputs[var_name]:
                    tensor = core.LoDTensor()
                    if isinstance(np_value, tuple):
                        tensor.set(np_value[0], self.place)
                        tensor.set_lod(np_value[1])
                    else:
                        tensor.set(np_value, self.place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(real_inputs[var_name], tuple):
                    tensor.set(real_inputs[var_name][0], self.place)
                    tensor.set_lod(real_inputs[var_name][1])
                else:
                    tensor.set(real_inputs[var_name], self.place)
                feed_map[var_name] = tensor

        return feed_map


    def run(self,input,output,attrs):
        program = Program()
        block = program.global_block()
        op_proto=self.op_proto
        inputs = self.__append_input_output(block, op_proto, input, True)
        outputs = self.__append_input_output(block, op_proto, output, False)
        op = block.append_op(
            type=self.op_name,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)
        fetch_list = []
        for var_name, var in outputs.iteritems():
            if var_name in outputs:
                if isinstance(var, list):
                    for v in var:
                        fetch_list.append(v)
                else:
                    fetch_list.append(var)
        feed_map = self.__feed_var(inputs, input)

        exe = Executor(self.place)
        result = exe.run(program,
                         feed=feed_map,
                         fetch_list=fetch_list,
                         return_numpy=False)


        atcual_dic = {}
        for i, obj in enumerate(fetch_list):
            atcual_dic[obj.name] = np.array(result[i])
        return atcual_dic

class SeqReader():
    def __init__(self,path):
        self.file=open(path,'r')
    def read(self,line=-1,line_skip=0,splite_sym=' '):
        if line==-1:
            lines=self.file.readlines()
        else:
            lines=self.file.readlines()[:line+1]
        ans=[]
        for one_line in lines:
            one_line=one_line[line_skip:]
            ans.append([float(x) for x in  one_line.split(splite_sym)])
        return ans


if __name__ == "__main__":
# test modeltest
#     reader=SeqReader('')
    #ModelPath = '/home/qa_work/CI/workspace/sys_anakin_compare_output/language/fluid_models'
    ModelPath = sys.argv[1]
    print(sys.argv[1])
    model = FluidModelTest(ModelPath,False)
    model.show()
   
    with open( sys.argv[2] ) as file:
    #with open('/home/qa_work/CI/workspace/sys_anakin_compare_output/language/input_file/fake_realtitle.test') as file:
        for i,line in enumerate(file.readlines()):
            print(line)
            inp=[int(k) for k in line.split(' ')]
            result=model.run([inp])
            np.save(sys.argv[3] + str(i), np.array(result[sys.argv[4]]))
            #np.save('/home/qa_work/CI/workspace/sys_anakin_compare_output/language/paddle_output/' +str(i), np.array(result['fc_3.tmp_2']))

    print(result.keys())

    exit()
