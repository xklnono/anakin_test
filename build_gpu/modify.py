#!/bin/python
import os
import sys
import shutil
import re
import fileinput

model = {
    #adu modle
    #'yolo': ['480', '1440', '3', 'public_images', 'cls_pred_out', 'obj_pred_out', 'ori_pred_out', 'dim_pred_out', 'loc_pred_out'],
    'yolo_lane_v2':         ['384', '960', '3', 'adu_images', 'softmax_out'],
    'yolo_camera_detector': ['480', '1440', '3', 'adu_images', 'cls_pred_out', 'obj_pred_out', 'ori_pred_out', 'dim_pred_out', 'loc_pred_out'],
    'cnn_seg':['8', '640', '640', 'adu_images', 'category_score_out', 'instance_pt_out', 'confidence_score_out', 'class_score_out', 'heading_pt_out', 'height_pt_out'],
    'diepsie': ['4', '256', '256', 'adu_images', 'cam_coords_out'],
    'densenbox': ['3', '512', '1600', 'adu_images', 'conv-out-scale-1_out'],
    #face modle
    'attribute': ['3', '144', '144', 'face_images', 'age_classification_prob_out', 'age_out', 'glasses_prob_out', 'expression_prob_out', 'race_prob_out', 'gender_prob_out'],
    'blur': ['3', '112', '112', 'face_images', 'prob_out'],
    'feature_patch0': ['3', '108', '108', 'face_images', 'patch_0_fc1_out'],
    'feature_patch1': ['3', '108', '108', 'face_images', 'patch_1_fc1_out'],
    'feature_patch2': ['3', '108', '108', 'face_images', 'patch_2_fc1_out'],
    'feature_patch3': ['3', '108', '108', 'face_images', 'patch_3_fc1_out'],
    'feature_patch4': ['3', '108', '108', 'face_images', 'patch_4_fc1_out'],
    'feature_patch5': ['3', '108', '108', 'face_images', 'patch_5_fc1_out'],
    'feature_patch6': ['3', '108', '108', 'face_images', 'patch_6_fc1_out'],
    'remark_demark':  ['3', '207', '175', 'face_images', 'tanh7_out'],
    'remark_super':   ['3', '207', '175', 'face_images', 'tanh7_out'],
    'score': ['216', '16', '16', 'face_images', 'prob_out'],
    'alignment_stage1': ['3', '112', '112', 'face_images', 'landmark_score_out'],
    'alignment_stage2': ['3', '144', '144', 'face_images', 'pose_predict_out'],
    'occlusion' : ['3', '64', '64', 'face_images', 'deconv8_out'],
    'liveness' : ['3', '224', '224', 'face_images', 'prob_out'],
    #vis modle  has no src and output
    #'mobilenet_ssd_caffe' : ['3', '300', '300', 'adu_images'],
    #'mobilenet_ssd_fluid' : ['3', '300', '300', 'adu_images'],
    #'se_resnext50_caffe' : ['3', '224', '224', 'adu_images'],
    #'se_resnext50_fluid' : ['3', '224', '224', 'adu_images'],
    #'ocr20' : ['0', '0', '0', 'adu_images'],
     #public modle
    'vgg16' : ['3', '224', '224', 'public_images', 'fc8_out'],
    'inception' : ['3', '224', '224', 'public_images', 'softmax_out'],
    'Resnet101' : ['3', '224', '224', 'public_images', 'prob_out'],
    'Resnet50' : ['3', '224', '224', 'public_images', 'prob_out'],
    'ssd' : ['3', '300', '300', 'public_images', 'detection_out_out'],
    'mobilenet' : ['3', '224', '224', 'public_images', 'prob_out'],
    'vgg19' : ['3', '224', '224', 'public_images', 'prob_out'],
    #'segnet' : ['3', '224', '224', 'public_images', 'prob_out'],
    'mobilenet_v2' : ['3', '224', '224', 'public_images', 'prob_out'],
    'yolo': ['3', '448', '448', 'public_images', 'result_out'],
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
                if "remark" in k:
                    if "cpu_data[idx] = img.data[w]" in line:
                        line = line.replace("cpu_data[idx] = img.data[w]","cpu_data[idx] = (img.data[w]-127.5)/127.5")
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

def modify_tensorRT_file(filename):
    cur_path = os.getcwd()
    src_path = os.path.join(cur_path, "new_file")
    src_name = filename
    print(src_name)
    src_abs = os.path.join(src_path, src_name)
    print(src_abs)  
    tmp_file_name = "tmp.cpp"
    tmp_abs = os.path.join(src_path, tmp_file_name)
    work_path = "/home/qa_work/CI/workspace/sys_tensorRT_merge_build"
    dst_path = os.path.join(work_path, "anakin1/adu-inference/test")
    if os.path.exists(src_abs) == False:
        print("file not exit")
        sys.exit(1)
    print(src_abs)
    num = len(sys.argv)
    for k in model:
        dst_name = os.path.join(dst_path, src_name.split('.')[0] + "_" + k + ".cpp")
        print(dst_name)
        shutil.copyfile(src_abs, tmp_abs)
        file_data=""
        with open(tmp_abs,"r")as f:
            for line in f:
                if "remark" in k:
                    if "cpu_data[idx] = img.data[w]" in line:
                        line = line.replace("cpu_data[idx] = img.data[w]","cpu_data[idx] = (img.data[w]-127.5)/127.5")
                if "model_name" in line:
                    line = line.replace("model_name", k)
                if "src_name" in line:
                    line = line.replace("src_name", model[k][3])
                if num == 3:
                    if sys.argv[2] == "k1200" :
                        if "TensorRT_time_p4.txt" in line:
                            line = line.replace("p4", "k1200")
                        if "DEFINE_GLOBAL(int, gpu, 0);" in line:
                            line = line.replace("0", "2");
            #print(tmp_out_data)
                file_data += line
        with open(tmp_abs, "w") as f:
            f.write(file_data)
        shutil.move(tmp_abs,dst_name)


###modify fuction && multithread && performace file
if sys.argv[1] == "anakin_p4":
    modify_anakin_file("model_batchsize_test.cpp", "p4")
    #modify_anakin_file("multi_test.cpp", "p4")
    modify_anakin_file("model_batchsize_test_perf.cpp", "p4")
    #modify_anakin_file("model_batchsize_test.cpp", "default")
    #modify_anakin_file("multi_test.cpp", "default")
    #modify_anakin_file("model_batchsize_test_perf.cpp", "default")
elif sys.argv[1] == "anakin_k1200":
    modify_anakin_file("model_batchsize_test.cpp", "k1200")
    modify_anakin_file("model_batchsize_test_perf.cpp", "k1200")
elif sys.argv[1] == "tensorRT":
    modify_tensorRT_file("rt_batchsize_test.cpp")   
    modify_tensorRT_file("rt_batchsize_test_perf.cpp")   
