#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "saber/core/tensor_op.h"
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <map>
#include <fstream>
#include <vector>
#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(std::string, model_dir, "");
DEFINE_GLOBAL(int, num, 1);
DEFINE_GLOBAL(int, channel, 8);
DEFINE_GLOBAL(int, height, 640);
DEFINE_GLOBAL(int, width, 640);
DEFINE_GLOBAL(bool, is_input_shape, false);
std::string img_path="";

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

void getModels(std::string path, std::vector<std::string>& files) {
    DIR* dir= nullptr;
    struct dirent* ptr;

    if ((dir = opendir(path.c_str())) == NULL) {
        perror("Open dri error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        } else if (ptr->d_type == 8) { //file
            files.push_back(path + "/" + ptr->d_name);
        } else if (ptr->d_type == 4) {
            //files.push_back(ptr->d_name);//dir
            getModels(path + "/" + ptr->d_name, files);
        }
    }

    closedir(dir);
}
TEST(NetTest, net_execute_base_test) {
    std::vector<std::string> models;
    getModels(GLB_model_dir, models);

    for (auto iter = models.begin(); iter < models.end(); iter++) {
        LOG(WARNING) << "load anakin model file from " << *iter << " ...";
#if 1
        Graph<NV, AK_FLOAT, Precision::FP32> graph;
        auto status = graph.load(*iter);
        

        if (!status) {
            LOG(FATAL) << " [ERROR] " << status.info();
        }

        if (GLB_is_input_shape) {
            graph.Reshape("input_0", {GLB_num, GLB_channel, GLB_height, GLB_width});
        } else {
            graph.ResetBatchSize("input_0", GLB_num);
        }
		graph.RegistOut("ft_add_left_right_out", "ft_add_left_right_out");
        graph.Optimize();
		
        // constructs the executer net
        Net<NV, AK_FLOAT, Precision::FP32> net_executer(graph, true);
        // get in
        auto d_tensor_in_p = net_executer.get_in("input_0");
        Tensor4d<X86, AK_FLOAT> h_tensor_in;
        auto valid_shape_in = d_tensor_in_p->valid_shape();

        for (int i = 0; i < valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        float* h_data = h_tensor_in.mutable_data();
        std::vector<float> img_data;
        int res = -1;
        if (img_path != "") {
            res = read_file(img_data, img_path.c_str());
            LOG(INFO)<<"res = "<<res;
        }
        if (res != -1) {
            for(int i = 0; i < h_tensor_in.size(); i++){
                //h_tensor_in.mutable_data()[i] = (float)img_data[i];
                h_data[i] = (float)img_data[i];
            }   
        } else {
            LOG(INFO)<<"NOT FOUND IMAGE!!!!!!!!!";
            fill_tensor_host_const(h_tensor_in, 1.0f);
        }
		//fill_tensor_host_rand(h_tensor_in, -1.0f, 1.0f);
		//for (int i = 0; i < h_tensor_in.size(); ++i) 
		//	printf ("%f\n", h_tensor_in.data()[i]);
        d_tensor_in_p->copy_from(h_tensor_in);
        //for diepise model
        auto d_tensor_in_1_p = net_executer.get_in("input_1");
        Tensor4d<X86, AK_FLOAT> h_tensor_in_1;

        h_tensor_in_1.re_alloc(d_tensor_in_1_p->valid_shape());
        for (int i=0; i<d_tensor_in_1_p->valid_shape().size(); i++) {
            LOG(INFO) << "detect input_1 dims[" << i << "]" << d_tensor_in_1_p->valid_shape()[i];
        }
    	h_data = h_tensor_in_1.mutable_data();
    	h_data[0] = 1408;
    	h_data[1] = 800;
    	h_data[2] = 0.733333;
    	h_data[3] = 0.733333;
    	h_data[4] = 0;
    	h_data[5] = 0;
    	d_tensor_in_1_p->copy_from(h_tensor_in_1);

    	auto d_tensor_in_2_p = net_executer.get_in("input_2");
    	Tensor4d<X86, AK_FLOAT> h_tensor_in_2;

    	h_tensor_in_2.re_alloc(d_tensor_in_2_p->valid_shape());
    	for (int i=0; i<d_tensor_in_2_p->valid_shape().size(); i++) {
        	LOG(INFO) << "detect input_2 dims[" << i << "]" << d_tensor_in_2_p->valid_shape()[i];
    	}	
    	h_data = h_tensor_in_2.mutable_data();
    	h_data[0] = 2022.56;
    	h_data[1] = 989.389;
    	h_data[2] = 2014.05;
    	h_data[3] = 570.615;
    	h_data[4] = 1.489;
		d_tensor_in_2_p->copy_from(h_tensor_in_2);
//
        int warmup_iter = 0;
        int epoch = 1;
        // do inference
        Context<NV> ctx(0, 0, 0);
        saber::SaberTimer<NV> my_time;
        LOG(WARNING) << "EXECUTER !!!!!!!! ";

        for (int i = 0; i < warmup_iter; i++) {
            net_executer.prediction();
        }

#ifdef ENABLE_OP_TIMER
        net_executer.reset_op_time();
#endif
        my_time.start(ctx);

        //auto start = std::chrono::system_clock::now();
        for (int i = 0; i < epoch; i++) {
            //DLOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
            net_executer.prediction();
        }
        cudaDeviceSynchronize();
        auto tp0 = net_executer.get_out("ft_add_left_right_out");
		//auto tp = net_executer.get_out("slice_[dump\, mask]_out");
		//auto tp1 = net_executer.get_out("category_score_out");
		//auto tp2 = net_executer.get_out("instance_pt_out");
		//auto tp3 = net_executer.get_out("confidence_score_out");
		//auto tp4 = net_executer.get_out("class_score_out");
		//auto tp5 = net_executer.get_out("heading_pt_out");
		//auto tp6 = net_executer.get_out("height_pt_out");
		Tensor<X86, AK_FLOAT, NCHW> host_tensor1;
		host_tensor1.re_alloc(tp0->valid_shape());
		host_tensor1.copy_from(*tp0);
		for (int i = 0; i < host_tensor1.size(); ++i)
			printf ("%f\n", host_tensor1.data()[i]);
        
        my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
        std::vector<float> op_time = net_executer.get_op_time();
        auto exec_funcs = net_executer.get_exec_funcs();
        auto op_param = net_executer.get_op_param();

        for (int i = 0; i <  op_time.size(); i++) {
            LOG(INFO) << "name: " << exec_funcs[i].name << " op_type: " << exec_funcs[i].op_name <<
                      " op_param: " << op_param[i] << " time " << op_time[i] / epoch;
        }

        std::map<std::string, float> op_map;

        for (int i = 0; i < op_time.size(); i++) {
            auto it = op_map.find(op_param[i]);

            if (it != op_map.end()) {
                op_map[op_param[i]] += op_time[i];
            } else {
                op_map.insert(std::pair<std::string, float>(op_param[i], op_time[i]));
            }
        }

        for (auto it = op_map.begin(); it != op_map.end(); ++it) {
            LOG(INFO) << it->first << "  " << (it->second) / epoch << " ms";
        }

#endif
        LOG(INFO) << *iter << " aveage time " << my_time.get_average_ms() / epoch << " ms";
        // save the optimized model to disk.
        //        std::string save_model_path = GLB_model_dir + std::string("opt.saved");
        //        status = graph.save(save_model_path);
        //        if (!status ) {
        //            LOG(FATAL) << " [ERROR] " << status.info();
        //        }
#endif
    }
}
int main(int argc, const char** argv) {
    // initial logger
    LOG(INFO) << "argc " << argc;

    if (argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/model_test\n \
            anakin_models\n \
            num\n \
            channel\n \
            height\n \
            width\n ";
        exit(0);
    } else if (argc == 2) {
        GLB_model_dir = std::string(argv[1]);
        GLB_is_input_shape = false;
    } else if (argc == 3) {
        GLB_model_dir = std::string(argv[1]);
        GLB_num = atoi(argv[2]);
        GLB_is_input_shape = false;
    } else {
        GLB_model_dir = std::string(argv[1]);
        GLB_num = atoi(argv[2]);
        GLB_channel = atoi(argv[3]);
        GLB_height = atoi(argv[4]);
        GLB_width = atoi(argv[5]);
        img_path = argv[6];
        GLB_is_input_shape = true;
    }

    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
