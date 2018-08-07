#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#define DEFINE_GLOBAL(type, var, value) \
		type (GLB_##var) = (value)

DEFINE_GLOBAL(int, gpu, 0);
DEFINE_GLOBAL(std::string, model, "");
DEFINE_GLOBAL(std::string, weights, "");
DEFINE_GLOBAL(std::string, img_path, "");

DEFINE_GLOBAL(int, offset_y, 0);
DEFINE_GLOBAL(bool, rgb, false);
DEFINE_GLOBAL(bool, vis, false);
DEFINE_GLOBAL(std::string, num, "1");
//DEFINE_GLOBAL(std::string, test_list, "/home/qa_work/CI/workspace/sys_anakin_compare_output/new256/input_file/batch" + GLB_num + "/file.list");
//DEFINE_GLOBAL(std::string, file_root, "/home/qa_work/CI/workspace/sys_anakin_compare_output/new256/input_file/batch" + GLB_num + "/");


// need to be changed
DEFINE_GLOBAL(std::string, image_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/mapdemo/images_output/");
DEFINE_GLOBAL(std::string, time_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/mapdemo/time/");
std::string model_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/mapdemo/models/demoprogram.anakin.bin";
std::string model_saved_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/mapdemo/models/";


void fill_image_data_yolo(const cv::Mat& img, float* cpu_data){
    int elem_num = img.channels() * img.rows * img.cols;
    //float * cpu_data = new float[elem_num];
    int idx = 0;
    float scale = 1.0f / 255;
    for(int c = 0; c < img.channels(); c++){
        for(int h = 0; h < img.rows; h++){
            for(int w = 0; w < img.cols; w++) {
                //cpu_data[idx] = img.data[idx] * scale;
                //cpu_data[idx] = img.data[idx];
                cpu_data[idx] = 1.0f;
                //cpu_data[idx] = std::rand()*1.0f/RAND_MAX -0.5;
                idx++;
             }
        }
    }
}

int read_file(std::vector<float> &results, const char* file_name) {

    std::ifstream infile(file_name);
    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return false;
    }
    LOG(INFO)<<"found filename: "<<file_name;
    std::string line;
    while (std::getline(infile, line)) {
        results.push_back((float)atof(line.c_str()));
    }
    return 0;
}

TEST(NetTest, net_execute_base_test) {
     Graph<NV, AK_FLOAT, Precision::FP32>* graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
     LOG(WARNING) << "load anakin model file from " << model_path << " ...";
     // load anakin model files.
     auto status = graph->load(model_path);
     if(!status ) {
         LOG(FATAL) << " [ERROR] " << status.info();
     }
    auto graph_ins = graph->get_ins();
    auto graph_outs = graph->get_outs();

    char * model_mapdemo = "mapdemo";
    if (std::strstr(model_path.c_str(), model_mapdemo)) {
        for (auto graph_in: graph_ins) {
            graph->ResetBatchSize(graph_in, std::stoi(GLB_num));
        }
        graph->Optimize();
        Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph, true);

        for (auto graph_in: graph_ins) {
    
            auto d_tensor_in_p = net_executer.get_in(graph_in);
            Tensor4d<X86, AK_FLOAT> h_tensor_in;
    
            auto valid_shape_in = d_tensor_in_p->valid_shape();
    
            h_tensor_in.re_alloc(valid_shape_in);
            float* h_data = h_tensor_in.mutable_data();
    
            for (int i = 0; i<h_tensor_in.size(); i++) {
                h_data[i] = 1.0f;
            }
    
            d_tensor_in_p->copy_from(h_tensor_in);
            cudaStreamSynchronize(NULL);
        }
        Context<NV> ctx(GLB_gpu, 0, 0);
        saber::SaberTimer<NV> my_time;
        LOG(WARNING) << "EXECUTER!!!!!!!";
        my_time.start(ctx);
        net_executer.prediction();
        cudaDeviceSynchronize();
        my_time.end(ctx);
        LOG(INFO) << "average time " << my_time.get_average_ms() << " ms";
        std::ofstream fout;
        std::string file;
        file = GLB_image_result + "mapdemp_output.txt" ;
        fout.open(file);
        auto tensor_out_0_p = net_executer.get_out(graph_outs[0]);
        test_print(tensor_out_0_p, fout);
        fout.close();
    }
    else {
        graph->ResetBatchSize("input_0", std::stoi(GLB_num));
        graph->Optimize();
   
        // constructs the executer net
        Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph, true);
        
        // get in
        auto d_tensor_in_p = net_executer.get_in("input_0");
        Tensor4d<X86, AK_FLOAT> h_tensor_in;

        auto valid_shape_in = d_tensor_in_p->valid_shape();

        std::ifstream fin;
        std::string test_list = "/home/qa_work/CI/workspace/sys_anakin_compare_output/new256/input_file/batch" + GLB_num + "/file.list";
        fin.open(test_list, std::ifstream::in);
        if(! fin.is_open()){
            LOG(ERROR) << "Failed to open test list file: " << test_list;
            return -1;
        }
        
        int img_num = 0;
        std::string file_name;
        while(fin >> file_name) {
            img_num++;
            //if (img_num > 1) {
            //   return 0;
           // }
            cudaDeviceSynchronize();
            std::string file_root = "/home/qa_work/CI/workspace/sys_anakin_compare_output/new256/input_file/batch" + GLB_num + "/";
            std::string file_path = file_root + file_name;

            for (int i=0; i<valid_shape_in.size(); i++) {
                LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
            }

            h_tensor_in.re_alloc(valid_shape_in);
            
            std::vector<float> input_data;
            int res = read_file(input_data, file_path.c_str());
            LOG(ERROR) << "h_tensor_in.size: " << h_tensor_in.size();
            LOG(ERROR) << "input_data.length: " << input_data.size();
            for (int i=0;i<h_tensor_in.size();i++){
	    		h_tensor_in.mutable_data()[i] = input_data[i];
                //h_tensor_in.mutable_data()[i] = 1.0f; 
            }

            LOG(ERROR) << "h_tensor_in.valid_size(): " << h_tensor_in.valid_size();
            d_tensor_in_p->copy_from(h_tensor_in);
             
            // get out-------begining
            Context<NV> ctx(GLB_gpu, 0, 0);
            saber::SaberTimer<NV> my_time;
            LOG(WARNING) << "EXECUTER!!!!!!!";
            my_time.start(ctx);

            net_executer.prediction();
            cudaDeviceSynchronize();
            my_time.end(ctx);
 
            cudaDeviceSynchronize();
            LOG(INFO) << "average time " << my_time.get_average_ms() << " ms";
            
            std::ofstream fout;
            std::string file;
            file = GLB_image_result + file_name + ".txt" ;
            fout.open(file);    
            char * model_new256 = "new256";
            for(int i = 0; i < graph_outs.size(); i++) {
                LOG(INFO) << "graph outs: " << graph_outs[i];
                auto tensor_out_i_p = net_executer.get_out(graph_outs[i]);
                if (std::strstr(model_path.c_str(), model_new256)) {
                    if (i==2) {
                        test_print(tensor_out_i_p, fout);
                    } else {
                        test_print_tmp(tensor_out_i_p, fout);
                    }
                }
                else {
                    test_print(tensor_out_i_p, fout); 
                }
            } 
            fout.close();
            // get out-------ending
	    }   
    }
}

int main(int argc, const char** argv){
    LOG(INFO) << "argc:" << argc;
    if(argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/model_batchsize_test_modelname\n \
        num \n";
    exit(0);
    } else if (argc == 2) {
        GLB_num = argv[1];
        LOG(INFO) << "default batch_size is : " << GLB_num;
    } else if (argc == 3) {
        GLB_img_path = argv[1];
        GLB_num = argv[2];
        LOG(INFO) << "Current batch_size is : " << GLB_num;
    }
	//cudaSetDevice(GLB_gpu);
    anakin::saber::Env<NV>::env_init();
    cudaSetDevice(GLB_gpu);
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
