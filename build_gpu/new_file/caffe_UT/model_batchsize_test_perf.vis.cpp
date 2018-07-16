#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

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
DEFINE_GLOBAL(int, num, 1);
DEFINE_GLOBAL(std::string, test_list, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/src_name/file.list");
DEFINE_GLOBAL(std::string, file_root, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/src_name/");

// need to be changed
DEFINE_GLOBAL(std::string, image_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/images_output/");
DEFINE_GLOBAL(std::string, time_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/time/");
std::string model_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/models/model_name.anakin.bin";
std::string model_saved_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/models/";


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

    graph->ResetBatchSize("input_0", GLB_num);
    graph->Optimize();
 
    auto graph_ins = graph->get_ins();
    auto graph_outs = graph->get_outs();   
 
    // constructs the executer net
    Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph, true);
    
    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86, AK_FLOAT> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();

    int width = d_tensor_in_p->height();
    int height = d_tensor_in_p->channel();

    std::ifstream fin;
    fin.open(GLB_test_list, std::ifstream::in);
    if(! fin.is_open()){
        LOG(ERROR) << "Failed to open test list file: " << GLB_test_list;
        return -1;
    }
    
    int img_num = 0;
    float t = 0.0f;
    float t_a = 0.0f;
    std::string file_name;
    while(fin >> file_name) {
        img_num++;
        //if (img_num > 1) {
        //   return 0;
       // }
        cudaDeviceSynchronize();
    
        std::string file_path = GLB_file_root + file_name;

        for (int i=0; i<valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        
        std::vector<float> input_data;
        int res = read_file(input_data, file_path.c_str());
        LOG(ERROR) << "h_tensor_in.size: " << h_tensor_in.size();
        LOG(ERROR) << "inpu_data.length: " << input_data.size();
        for (int i=0;i<h_tensor_in.size();i++){
			h_tensor_in.mutable_data()[i] = input_data[i];
            //h_tensor_in.mutable_data()[i] = 1.0f; 
        }

        LOG(ERROR) << "h_tensor_in.valid_size(): " << h_tensor_in.valid_size();
        d_tensor_in_p->copy_from(h_tensor_in);

        // get out-------begining
        net_executer.prediction();
        cudaDeviceSynchronize();
        
        std::ofstream fout;
        std::string file;
        file = GLB_image_result + file_name + ".txt" ;
        fout.open(file);    
        
        //std::istringstream str( tmp_out ); 
        //std::vector<std::string> out_name ;
        //std::string tmp;
        //while (std::getline(str, tmp, ';')) {
        //    out_name.push_back(tmp); 
        //}
        //for(int i = 0; i < out_name.size(); i++) {
        //    auto tensor_out_i_p = net_executer.get_out(out_name[i]);
        //    test_print(tensor_out_i_p, fout);
        //}
        
        for(int i = 0; i < graph_outs.size(); i++) {
            LOG(INFO) << "graph outs: " << graph_outs[i];
            auto tensor_out_i_p = net_executer.get_out(graph_outs[i]);
            test_print(tensor_out_i_p, fout);
        }
  
        fout.close();
        // get out-------ending
        int epoch = 1000;
        Context<NV> ctx(GLB_gpu, 0, 0);
        saber::SaberTimer<NV> my_time;
        LOG(WARNING) << "EXECUTER!!!!!!!";
        my_time.start(ctx);
        for(int i = 0; i < epoch; i++) {
            net_executer.prediction();
        }
        my_time.end(ctx);
        LOG(INFO) << "average time " << my_time.get_average_ms()/epoch << " ms";
        t += my_time.get_average_ms()/epoch;
	}   
    LOG(INFO) << "file num is : " << img_num ;
    //save time to file
    std::ofstream fp;
    fp.open(GLB_time_result + "Anakin2_time" + ".txt");
    LOG(INFO) << "image num: "<< img_num;
    t_a = t / img_num;
    LOG(INFO) << "total time: " << t << "ms" ;
    LOG(INFO) << "average time: " << t_a << "ms";
    fp << "image_num : " << img_num << std::endl;
    fp << "total_time : " << t << "ms" << std::endl;
    fp << "average_time : " << t_a << "ms" << std::endl;
    fp.close();
}

int main(int argc, const char** argv){
    LOG(INFO) << "argc:" << argc;
    if(argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/model_batchsize_test_modelname\n \
        num \n";
    exit(0);
    } else if (argc == 2) {
        GLB_num = atoi(argv[1]);
        LOG(INFO) << "default batch_size is : " << GLB_num;
    } else if (argc == 3) {
        GLB_img_path = argv[1];
        GLB_num = atoi(argv[2]);
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
