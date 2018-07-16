#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#define DEFINE_GLOBAL(type, var, value) \
		type (GLB_##var) = (value)

DEFINE_GLOBAL(int, gpu, 0);
DEFINE_GLOBAL(std::string, model, "");
DEFINE_GLOBAL(std::string, weights, "");

DEFINE_GLOBAL(int, offset_y, 0);
DEFINE_GLOBAL(bool, rgb, false);
DEFINE_GLOBAL(bool, vis, false);
DEFINE_GLOBAL(int, num, 1);
DEFINE_GLOBAL(std::string, test_list, "/home/qa_work/CI/workspace/sys_anakin_compare_output/src_name/tail_10.list");
DEFINE_GLOBAL(std::string, image_root, "/home/qa_work/CI/workspace/sys_anakin_compare_output/src_name/");

// need to be changed
DEFINE_GLOBAL(std::string, image_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/images_output/");
DEFINE_GLOBAL(std::string, time_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/time/");
std::string model_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/models/model_name.anakin.bin";
std::string model_saved_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/models/";


void fill_image_data_yolo(const cv::Mat& img, float* cpu_data, int num){
    int elem_num = img.channels() * img.rows * img.cols * num;
    //float * cpu_data = new float[elem_num];
    int idx = 0;
    float scale = 1.0f / 255;
    int img_size = img.channels() * img.rows * img.cols;
    for(int n = 0; n < num; n++){
        for(int w = 0; w < img_size; w++) {
            //cpu_data[idx] = img.data[idx] * scale;
            //cpu_data[idx] = img.data[idx];
            cpu_data[idx] = img.data[w];
            //cpu_data[idx] = 1.0f;
            //cpu_data[idx] = std::rand()*1.0f/RAND_MAX -0.5;
            idx++;
         }
    }
   // cudaMemcpy((void*)gpu_data, (void*)cpu_data, elem_num* sizeof(float), cudaMemcpyHostToDevice);

   // delete[]  cpu_data;
}

TEST(NetTest, net_execute_base_test) {
     Graph<NV, AK_FLOAT, Precision::FP32>* graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
     LOG(WARNING) << "load anakin model file from " << model_path << " ...";
     // load anakin model files.
     auto status = graph->load(model_path);
     if(!status ) {
         LOG(FATAL) << " [ERROR] " << status.info();
     }

    //reshape shape batch-size
    //graph->Reshape("input_0", {1, 8, 640, 640});
    //graph->Reshape("input_0", {GLB_num, userdefined_channel, userdefined_height, userdefined_width});
    graph->ResetBatchSize("input_0", GLB_num);
//graph->RegistAllOut();
    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph, true);
    
    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86, AK_FLOAT> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();

    int width = d_tensor_in_p->width();
    int height = d_tensor_in_p->height();
    int num = d_tensor_in_p->num();

    std::ifstream fin;
    fin.open(GLB_test_list, std::ifstream::in);
    if(! fin.is_open()){
        LOG(ERROR) << "Failed to open test list file: " << GLB_test_list;
        return -1;
    }
    
    int img_num = 0;
    float t = 0.0f;
    float t_a = 0.0f;
    std::string image_name;
    while(fin >> image_name) {
        img_num++;
       // if (img_num > 1) {
       //    return 0;
       // }
        cudaDeviceSynchronize();
    
        std::string image_path = GLB_image_root + image_name;
        cv::Mat img = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
        cv::Mat img_org;
        img.copyTo(img_org);

        cv::Rect roi(0, GLB_offset_y, img.cols, img.rows - GLB_offset_y);
        cv::Mat img_roi = img(roi);
        img_roi.copyTo(img);

        if (img.data == 0) {
            LOG(ERROR) << "Failed to read iamge: " << image_path;
            return -1;
        }

        int org_width = img.cols;
        int org_height = img.rows;
        cv::resize(img, img, cv::Size(width, height));
        if (GLB_rgb) {
            cv::cvtColor(img, img, CV_BGR2RGB);
        }

        for (int i=0; i<valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        float* h_data = h_tensor_in.mutable_data();

        fill_image_data_yolo(img, h_data, num);

        d_tensor_in_p->copy_from(h_tensor_in);

        // get out-------begining
        Context<NV> ctx(GLB_gpu, 0, 0);
        saber::SaberTimer<NV> my_time;
        LOG(INFO) << "EXECUTER!!!!!";
        my_time.start(ctx);

        net_executer.prediction();
        cudaDeviceSynchronize();
        my_time.end(ctx);
        LOG(INFO) << "one epoch time is: " << my_time.get_average_ms() << " ms";
        //write info file        
        std::ofstream fout;
        std::string file;
        file = GLB_image_result + image_name + ".txt" ;
        fout.open(file);    
        
        std::istringstream str( tmp_out ); 
        std::vector<std::string> out_name ;
        std::string tmp;
        while (std::getline(str, tmp, ';')) {
            out_name.push_back(tmp); 
        }
        for(int i = 0; i < out_name.size(); i++) {
            auto tensor_out_i_p = net_executer.get_out(out_name[i]);
            test_print(tensor_out_i_p, fout);
        }
   
        fout.close();
        // get out-------ending

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
        GLB_num = atoi(argv[1]);
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
