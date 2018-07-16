#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <iostream>
#include <fstream>

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
DEFINE_GLOBAL(std::string, test_list, "/home/qa_work/CI/workspace/sys_anakin_compare_output/src_name/tail.list");
DEFINE_GLOBAL(std::string, image_root, "/home/qa_work/CI/workspace/sys_anakin_compare_output/src_name/");

// need to be changed
DEFINE_GLOBAL(std::string, image_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/images_output/");
DEFINE_GLOBAL(std::string, time_result, "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/multi_thread_time/");
std::string model_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/models/model_name.anakin.bin";
std::string model_saved_path = "/home/qa_work/CI/workspace/sys_anakin_compare_output/model_name/models/";
DEFINE_GLOBAL(int, thread_num, 1);
DEFINE_GLOBAL(int, batch_size, 1);

void fill_image_data_yolo(const cv::Mat& img, float* cpu_data){
    int elem_num = img.channels() * img.rows * img.cols;
    //float * cpu_data = new float[elem_num];
    int idx = 0;
    float scale = 1.0f / 255;
    for(int c = 0; c < img.channels(); c++){
        for(int h = 0; h < img.rows; h++){
            for(int w = 0; w < img.cols; w++) {
                //cpu_data[idx] = img.data[idx] * scale;
                cpu_data[idx] = img.data[idx];
                //cpu_data[idx] = 0;
                //cpu_data[idx] = std::rand()*1.0f/RAND_MAX -0.5;
                idx++;
             }
        }
    }
   // cudaMemcpy((void*)gpu_data, (void*)cpu_data, elem_num* sizeof(float), cudaMemcpyHostToDevice);

   // delete[]  cpu_data;
}

TEST(NetTest, net_execute_sync_multithread_test) {
    Env<NV>::env_init(1);
    LOG(INFO) << "Sync Runing multi_threads for model:" << model_path;
    Worker<NV, AK_FLOAT, Precision::FP32> workers(model_path, GLB_thread_num);
    workers.register_inputs({"input_0"});
   
    std::istringstream str( tmp_out );
    std::vector<std::string> out_name ;
    std::string tmp;
    while (std::getline(str, tmp, ';')) {
        out_name.push_back(tmp);
    }
    for(int i = 0; i < out_name.size(); i++) {
        workers.register_outputs({out_name[i]});
    }
    
    workers.Reshape("input_0",{GLB_batch_size, userdefined_channel, userdefined_height, userdefined_width});
    //workers.ResetBatchSize("input_0", GLB_batch_size);
    workers.launch(); 
   
    std::vector<Tensor4dPtr<target_host<NV>::type, AK_FLOAT> > host_tensor_p_in_list;
 
    // get in
    saber::Shape valid_shape_in({GLB_batch_size, userdefined_channel, userdefined_height, userdefined_width});
    Tensor4dPtr<target_host<NV>::type, AK_FLOAT> h_tensor_in = new Tensor4d<target_host<NV>::type, AK_FLOAT>(valid_shape_in);
    int width = h_tensor_in->height();
    int height = h_tensor_in->channel();

    std::ifstream fin;
    fin.open(GLB_test_list, std::ifstream::in);
    if(! fin.is_open()){
        LOG(ERROR) << "Failed to open test list file: " << GLB_test_list;
        return -1;
    }
    
    int img_num = 0;
    float t = 0.0f;
    float t_a = 0.0f;
    float qps = 0.0f;
    std::string image_name;
    while(fin >> image_name) {
        img_num++;
       // if (img_num > 1) {
       //    return 0;
       // }
    
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

        for (int i=0; i< valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }
        //h_tensor_in->re_alloc(valid_shape_in);
        float* h_data = h_tensor_in->mutable_data();
      
        fill_image_data_yolo(img, h_data);
        
        host_tensor_p_in_list.push_back(h_tensor_in);

        int epoch = 10;
        
        //Context<NV> ctx(0, 0, 0);
        //saber::SaberTimer<NV> my_time; 
   		//my_time.start(ctx);
        for(int i=0; i< epoch; i++) {
            auto d_tensor_p_out_list = workers.sync_prediction(host_tensor_p_in_list);
        }
   		//my_time.end(ctx);
        //LOG(INFO)<<"aveage time "<<my_time.get_average_ms() << " ms";
#ifdef ENABLE_OP_TIMER
       auto& times_map = workers.get_task_exe_times_map_of_sync_api();
       for(auto it = times_map.begin(); it != times_map.end(); it++) {
           LOG(WARNING) << " threadId: " << it->first << " processing " << it->second.size() << " tasks";
           for(auto time_in_ms : it->second) {
               LOG(INFO) << "    \\__task avg time: " << time_in_ms;
               t += time_in_ms;
           }
       }
       qps = (img_num * 1000.0)/ t ;
       //t_a = t / GLB_thread_num;
       LOG(INFO) << "total time is : " << t << "ms" ;
       LOG(INFO) << "img_num : " << img_num ;
       LOG(INFO) << "qps is : " << qps  ;
#endif
       
       // std::ofstream fout;
       // std::string file;
       // file =  GLB_image_result + image_name + ".txttxt" ;
       // fout.open(file);
       // test_print(d_tensor_p, fout);
       // fout.close();

    }
    std::ofstream fp;
    if( GLB_batch_size == 1 && GLB_thread_num ==1 ){
         fp.open(GLB_time_result + "Multi_thread_time.txt");
    } else {
         fp.open(GLB_time_result + "Multi_thread_time.txt", std::ios::app);
    }
    LOG(INFO) << "batch_size is : " << GLB_batch_size;
    LOG(INFO) << "thread_num is : " << GLB_thread_num;
    LOG(INFO) << "qps is :  " << qps << "ms";

    fp << "batch_size is : " << GLB_batch_size << " thread_num is : " << GLB_thread_num  << " qps is : " << qps << std::endl;
    fp.close();
    fin.clear();
}

//TEST(NetTest, net_execute_async_multithread_test) {
//    LOG(INFO) << "Async Runing multi_threads for model:" << model_path;
//    Worker<NV, AK_FLOAT, Precision::FP32> workers(model_path, GLB_thread_num);
//    workers.register_inputs({"input_0"});
//    workers.register_outputs({"softmax_out"});
//    workers.Reshape("input_0",{GLB_batch_size, 384, 960, 3});
//    workers.launch(); 
//   
//    std::vector<Tensor4dPtr<X86, AK_FLOAT> > host_tensor_p_in_list;
// 
//    // get in
//    saber::Shape valid_shape_in({GLB_batch_size, 384, 960, 3});
//    Tensor4dPtr<X86, AK_FLOAT> h_tensor_in = new Tensor4d<X86, AK_FLOAT>(valid_shape_in);
//
//    int width = h_tensor_in->height();
//    int height = h_tensor_in->channel();
//
//    std::ifstream fin;
//    fin.open(GLB_test_list, std::ifstream::in);
//    if(! fin.is_open()){
//        LOG(ERROR) << "Failed to open test list file: " << GLB_test_list;
//        return -1;
//    }
//    
//    int img_num = 0;
//    float t = 0.0f;
//    float t_a = 0.0f;
//    std::string image_name;
//    while(fin >> image_name) {
//        img_num++;
//       // if (img_num > 1) {
//       //    return 0;
//       // }
//    
//        std::string image_path = GLB_image_root + image_name;
//        cv::Mat img = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
//        cv::Mat img_org;
//        img.copyTo(img_org);
//
//        cv::Rect roi(0, GLB_offset_y, img.cols, img.rows - GLB_offset_y);
//        cv::Mat img_roi = img(roi);
//        img_roi.copyTo(img);
//
//        if (img.data == 0) {
//            LOG(ERROR) << "Failed to read iamge: " << image_path;
//            return -1;
//        }
//
//        int org_width = img.cols;
//        int org_height = img.rows;
//        cv::resize(img, img, cv::Size(width, height));
//        if (GLB_rgb) {
//            cv::cvtColor(img, img, CV_BGR2RGB);
//        }
//
//        for (int i=0; i< valid_shape_in.size(); i++) {
//            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
//        }
//        //h_tensor_in->re_alloc(valid_shape_in);
//        float* h_data = h_tensor_in->mutable_data();
//      
//        fill_image_data_yolo(img, h_data);
//        
//        host_tensor_p_in_list.push_back(h_tensor_in);
//
//        int epoch = 1;
//        
//         
//        for(int i=0; i< epoch; i++) {
//            workers.async_prediction(host_tensor_p_in_list);    
//        }
//        int iterator = epoch;
//        while(iterator) {
//            if(!workers.empty()) {
//                auto d_tensor_p = workers.async_get_result()[0];
//                iterator--;
//            }
//        }
//       // std::ofstream fout;
//       // std::string file;
//       // file =  GLB_image_result + image_name + ".txttxt" ;
//       // fout.open(file);
//       // test_print(d_tensor_p, fout);
//       // fout.close();
//    }
//
//}

int main(int argc, const char** argv){
    // initial logger
    if (argc < 3) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/multi_test\n \
        batch_size\n \
        thread_num\n ";
        exit(0);
    } else {
        GLB_batch_size = atoi(argv[1]);
        GLB_thread_num = atoi(argv[2]);
    }
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
