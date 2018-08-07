#!/bin/python
import os
import sys
import shutil
import re
import fileinput

model = {
    #vis modle
    #'ocr20' : ['1', '48', '194', 'input_file', 'fc_2.tmp_out'],
    'mainbody' : ['3', '227', '227', 'input_file', 'elementwise_add_0.tmp_0_out'],
    'mobilenetssd' : ['3', '300', '300', 'input_file', 'detection_output_0.tmp_0_out'],
    'animal_v2' : ['3', '224', '224', 'input_file', 'fc_reduce.tmp_1_out'],
    'plant_v2' : ['3', '222', '222', 'input_file', 'fc_reduce.tmp_1_out'],
    'new256' : ['3', '225', '225', 'input_file','concat_0.tmp_0_out', 'concat_1.tmp_0_out', 'fc2.tmp_1_out'],
    'se_ResNeXt50' : ['3', '224', '224', 'input_file', 'fc_32.tmp_2_out'],
    'car_ssd' : ['3', '300', '300', 'input_file', 'detection_out'],
}


def modify_anakin_file(file_name, card_name):
    cur_path = os.getcwd()
    src_path = os.path.join(cur_path, "new_file")
    src_name = file_name
    card = card_name
    src_abs = os.path.join(src_path, src_name)
    tmp_file_name = "tmp.cpp"
    tmp_abs = os.path.join(src_path, tmp_file_name)
    if card == "k1200":
        work_path = "/home/qa_work/CI/workspace/sys_anakin_merge_build_k1200"
    else:
        work_path = "/home/qa_work/CI/workspace/sys_anakin_merge_build"
    dst_path = os.path.join(work_path, "anakin2/test/framework/net")
    if os.path.exists(src_abs) == False:
        print("file not exit")
        sys.exit(1)
    print(src_abs)
    for k in model:
        if card == "k1200":
            dst_name = os.path.join(dst_path, src_name.split('.')[0] + "_" + k + "_k1200.cpp")
        elif card == "p4":
            dst_name = os.path.join(dst_path, src_name.split('.')[0] + "_" + k + "_p4.cpp")
        else:
            dst_name = os.path.join(dst_path, src_name.split('.')[0] + "_" + k + ".cpp")
        print(dst_name)
        shutil.copyfile(src_abs, tmp_abs)
        file_data=""
        tmp_out_data="\""
        for i in range(len(model[k])-4):
            tmp_out_data += model[k][i+4] +';'
        tmp_out_data += "\""
        with open(tmp_abs,"r")as f:
            for line in f:
                if "model_name" in line:
                    line = line.replace("model_name", k)
                if "src_name" in line:
                    line = line.replace("src_name", model[k][3])
                if "tmp_out" in line:
                    line = line.replace("tmp_out", tmp_out_data)
                if file_name == "multi_test.cpp":
                    if "userdefined_channel" in line:
                        line = line.replace("userdefined_channel", model[k][0])
                    if "userdefined_height" in line:
                        line = line.replace("userdefined_height", model[k][1])
                    if "userdefined_width" in line:
                        line = line.replace("userdefined_width", model[k][2])
                if card == "p4":
                    if "Anakin2_time" in line:
                        line = line.replace(".txt", "_p4.txt");
                if card == "k1200":
                    if "DEFINE_GLOBAL(int, gpu, 0);" in line:
                        line = line.replace("0", "2")
                    if "Anakin2_time" in line:
                        line = line.replace(".txt", "_k1200.txt");
                file_data += line
        with open(tmp_abs, "w") as f:
            f.write(file_data)
        shutil.move(tmp_abs, dst_name)
###modify fuction && multithread && performace file
if sys.argv[1] == "anakin_p4":
    modify_anakin_file("model_batchsize_test.vis.cpp", "p4")
    #modify_anakin_file("multi_test.cpp", "p4")
    modify_anakin_file("model_batchsize_test_perf.vis.cpp", "p4")
    modify_anakin_file("model_batchsize_test.vis.cpp", "default")
    #modify_anakin_file("multi_test.cpp", "default")
    #modify_anakin_file("model_batchsize_test_perf.vis.cpp", "default")
elif sys.argv[1] == "anakin_k1200":
    modify_anakin_file("model_batchsize_test.vis.cpp", "k1200")
    modify_anakin_file("model_batchsize_test_perf.vis.cpp", "k1200")
