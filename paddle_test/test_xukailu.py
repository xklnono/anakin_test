import numpy as np
import paddle.fluid as fluid
import os
import sys
#from paddle.fluid import debuger
from datetime import datetime

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

    target_output_varname = 'elementwise_add_0.tmp_0'
    target_input_varname = 'data'
    num = 1
    channel = 3
    height = 227
    width = 227

    #TxtPath = '/home/qa_work/wgy/paddle_test/input_mainbody_0_1.txt'
    if len(sys.argv) < 2:
        print("Usage: please input the input.txt")
        sys.exit(1)
    TxtPath = sys.argv[1]
    ModelPath = '/home/qa_work/wgy/mainbody/fluid/'
    #Place = fluid.CUDAPlace(0)
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

        #############
        #fp = open('net_work.txt', 'w')
        #fp.write("%s" % net_program)
        #fp.close()
        #print(net_program)  #could print dimm nums, like n c h w
        global_block = net_program.global_block()
        source_ops = list(global_block.ops)
        debugger = Fluid_debugger()

        #fluid_feed = np.ones((num, channel, height, width), dtype=np.float32)
        a = np.loadtxt(TxtPath, dtype=np.float32)
        #a = np.ones((num, channel, height, width), dtype=np.float32)
        fluid_feed = np.reshape(a, [-1, channel, height, width])
        
        #debuger.draw_block_graphviz(global_block, path="./mobilessd_before.dot")
        fetch_list = debugger.fetch_tmp_vars(global_block, fetch_targets, [target_output_varname])
        t = 0
        idx = 0
        for var in fetch_list:
            if var.name == target_output_varname:
                var_idx = idx
            idx = idx + 1

        #######new#########
       # count = 0
      #  result = []
      #  for i in range(0,10):
      #      t1 = datetime.now()
            results = []
            results = Exe.run(program = net_program, feed = {target_input_varname: fluid_feed},
                            fetch_list = fetch_list, return_numpy = False)
      #      t2 = datetime.now()
      #      print 'run time is %f ms' % (t2-t1).total_seconds()
      #      t += (t2-t1).total_seconds()
      #      count += 1
      #  print 'run %d times, time is %f ms, average time is %f ms' % (count, t, t/count) 

        print np.array(results[var_idx])
        print np.array(results[var_idx]).shape

        #---------starting--------
        fetch_txt_fp = open('mainbody_output.txt', 'w')

        list_results = list(np.array(results[var_idx]))
        num = len(list_results)
        channel = 4
        height = 1
        width = 1

        fluid_fetch_list = list(np.array(results[var_idx]).flatten())
        for n in range(num):
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        item = fluid_fetch_list.pop(0)
                        fetch_txt_fp.write(str('%.18f' % item))
                    fetch_txt_fp.write(str('\n'))
                fetch_txt_fp.write(str('\n'))
            fetch_txt_fp.write(str('\n'))

        fetch_txt_fp.write('n = %d, ' % num)
        fetch_txt_fp.write('c = %d, ' % channel)
        fetch_txt_fp.write('h = %d, ' % height)
        fetch_txt_fp.write('w = %d' % width)

        fetch_txt_fp.close()
        #---------ending--------
