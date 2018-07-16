from __future__ import print_function
import numpy as np
import os
import sys

#np_dir_list= '/home/qa_work/CI/workspace/sys_anakin_compare_output/language/paddle_output'
#txt_dir_list= '/home/qa_work/CI/workspace/sys_anakin_merge_build/anakin2/tmp'
np_dir_list=sys.argv[1]
txt_dir_list=sys.argv[2]

np_list=os.listdir(np_dir_list)
txt_list=os.listdir(txt_dir_list)

def read_line_txt(path):
    with open(path) as file:
        k=[float(i.split(' ')[1]) for i in file.readlines()]
        return np.array(k)
inner_output={path[:-4]:np.load(np_dir_list + '/' + path) for path in np_list}
txt_output={path[:-4]:read_line_txt(txt_dir_list+ '/' + path)  for path in txt_list if path.startswith('output_')}

# print(txt_list)
for i in txt_output.keys():
    k=i[i.rfind('_')+1:]
    print(i)
    anakin=txt_output[i]
    fluid=inner_output[k]
    diff=anakin-fluid.flatten()
    if np.max(diff)>0.0001:
        print(k,np.max(diff))

# print(x[:10])
