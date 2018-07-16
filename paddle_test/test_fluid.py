import numpy as np
import paddle.fluid as fluid
import os
from paddle.fluid import debuger


class Fluid_debugger:

    def var_names_of_fetch(self, fetch_targets):
        var_names_list = []
        for var in fetch_targets:
            var_names_list.append(var.name)
        return var_names_list


    def fetch_tmp_vars(self, block, fetch_targets, var_names_list = None):
        fetch_var = block.var('fetch')
        old_fetch_names = self.var_names_of_fetch(fetch_targets)
        new_fetch_vars = []
        for var_name in old_fetch_names:
            var = block.var(var_name)
            new_fetch_vars.append(var)
        i = len(new_fetch_vars)
        if var_names_list is None:
            var_names_list = block.vars.keys()
        for var_name in var_names_list:
            if '.tmp_' in var_name and var_name not in old_fetch_names:
                print var_name
                var = block.var(var_name)
                new_fetch_vars.append(var)
                block.append_op(
                    type='fetch',
                    inputs={'X': [var_name]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})
                i = i + 1
        return new_fetch_vars



if __name__ == "__main__":

    target_output_varname = 'fc_2.tmp_4'
    target_input_varname = 'pixel'
    num = 1
    channel = 1
    height = 48
    width = 410

    TxtPath = '/home/qa_work/wgy/ocr20/inputs/48_410.txt'
    ModelPath = '/home/qa_work/wgy/ocr20/ocr_eng_20conv_baseline_fc/'
    Place = fluid.CPUPlace()
    Exe = fluid.Executor(Place)
    Scope = fluid.core.Scope()

    with fluid.scope_guard(Scope):
        if os.path.exists(ModelPath + 'model') and os.path.exists(ModelPath + 'params'):
            [net_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(ModelPath, Exe, 'model', 'params')
        else:
            [net_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(ModelPath, Exe)

        global_block = net_program.global_block()
        source_ops = list(global_block.ops)
        fluid_debugger = Fluid_debugger()

        #fluid_feed = np.ones((num, channel, height, width), dtype=np.float32)
        a = np.loadtxt(TxtPath, dtype=np.float32)
        fluid_feed = np.reshape(a, [-1, channel, height, width])

        #debugger.draw_block_graphviz(global_block, path="./mobilessd_before.dot")
        fetch_list = fluid_debugger.fetch_tmp_vars(global_block, fetch_targets, [target_output_varname])

        idx = 0
        for var in fetch_list:
            if var.name == target_output_varname:
                var_idx = idx
                print var_idx
            idx = idx + 1


        results = Exe.run(program = net_program, feed = {target_input_varname: fluid_feed},
                            fetch_list = fetch_list, return_numpy = False)
        print np.array(results[var_idx])
        print np.array(results[var_idx]).shape
        fluid_fetch_list = list(np.array(results[var_idx]).flatten())
        fetch_txt_fp = open('_output.txt', 'w')
        for num in fluid_fetch_list:
            fetch_txt_fp.write(str('%.18f' % num) + '\n')
        fetch_txt_fp.close()
