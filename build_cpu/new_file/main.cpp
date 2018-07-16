#include <string.h>
#include "stdio.h"
#include "stdlib.h"
#include <string>
#include <vector>
#include <pthread.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include "liblego/lego.h"

char* net_proto_file;
char* model_file;
char* input_file;
char* time_file;
namespace pairwise_demo{
    unsigned int g_thread_number = 2;
    const int g_batch_size = 1;
    const char* blob_name = "";
    std::vector<std::string> inputed_lines;
    float total_time = 0;
    
    // the global predictor
    liblego::Net g_net;
    void load_input_lines(char *filename) {
        static const int max_line_buf_size = 100 * 1024 * 1024;
        char *line_buffer = (char *)calloc(max_line_buf_size, sizeof(char));
        FILE *input_file = fopen(filename, "r");

        while (fgets(line_buffer, max_line_buf_size, input_file)) {
            // trim newline at end
            char *pos = NULL;
            if ((pos = strchr(line_buffer, '\n')) != NULL){
                *pos = 0;
            }
            inputed_lines.push_back(line_buffer);
        }
        free(line_buffer);
        line_buffer = NULL;
        fclose(input_file);
    }

    void split2(
            const std::string& main_str,
            std::vector<std::string>& str_list,
            const std::string & delimiter) {
        size_t pre_pos = 0;
        size_t position = 0;
        std::string tmp_str;

        str_list.clear();
        if (main_str.empty()) {
            return;
        }

        while ((position = main_str.find(delimiter, pre_pos)) != std::string::npos) {
            tmp_str.assign(main_str, pre_pos, position - pre_pos);
            str_list.push_back(tmp_str);
            pre_pos = position + 1;
        }

        tmp_str.assign(main_str, pre_pos, main_str.length() - pre_pos);

        if (!tmp_str.empty()) {
            str_list.push_back(tmp_str);
        }
    }

    int string_to_id_buffer(
            float* out_buffer, const int capacity, const std::string& str) {
        std::vector<std::string> id_strs;
        split2(str, id_strs, std::string(" "));
	//printf ("str is %s\n", str.c_str());
        if ((int)id_strs.size() > capacity){
            fprintf(stderr, "input length(%lu) is larger than capacity(%d)\n",
                    id_strs.size(), capacity);
            return -1;
        }
        for (size_t i = 0; i < id_strs.size(); i++){
            out_buffer[i] = static_cast<float>(atof(id_strs[i].c_str()));
        }
        return id_strs.size();
    }

    int string_to_input(const std::string& line, void* tlr_ptr){
        std::vector<std::string> number_strs;
        split2(line, number_strs, std::string(";"));

        std::vector<liblego::Blob*> input_blobs = g_net.input_blobs(tlr_ptr);
        int input_size = g_net.input_size();
        if ((int)number_strs.size() < input_size + 1){
            fprintf(stderr, "input slots is no enough, has %lu expect %d",
                    number_strs.size(), input_size);
            return -1;
        }
        for (int i = 0; i < input_size; i++){
            int ret =
                string_to_id_buffer(input_blobs[i]->mutable_data(),
                        input_blobs[i]->capacity, number_strs[i + 1]);
            if (ret == -1){
                return -1;
            }else{
                input_blobs[i]->dim0 = ret;
            }
        }
        return 0;
    }

    int batch_string_to_input(
            const std::vector<std::string> &line_vec, void *tlr_ptr){
        std::vector<liblego::Blob*> input_blobs = g_net.input_blobs(tlr_ptr);
        size_t input_size = g_net.input_size();
        std::vector<std::vector<int> > offset;
        offset.resize(input_size);
        int batch = line_vec.size();
        for (size_t i = 0; i < input_size; i++) {
            offset[i].resize(batch + 1);
            offset[i][0] = 0;
        }
        std::vector<std::string> number_strs;
        for (size_t i = 0; i < line_vec.size(); i++) {
            split2(line_vec[i], number_strs, std::string(";"));
            if (number_strs.size() < input_size + 1){
                fprintf(stderr, "input slots is no enough, has %lu expect %lu",
                        number_strs.size(), input_size);
                return -1;
            }
            for (size_t j = 0; j < input_size; j++) {
                if (i == 0) {
                    input_blobs[j]->dim0 = 0;
                }
                float * data_ptr = input_blobs[j]->mutable_data();
                int ret = string_to_id_buffer(
                                data_ptr + offset[j][i],
                                input_blobs[j]->capacity - offset[j][i], 
                                number_strs[j + 1]);
                if (ret == -1) {
                    return -1;
                } else {
                    input_blobs[j]->dim0 += ret;
                }
                offset[j][i + 1] = offset[j][i] + ret;
            }
        }
        for (size_t i = 0; i < input_size; i++) {
            g_net.set_input_blobs_batch_offset(tlr_ptr, offset[i], i);
        }
        return 0;
    }

   float diff_time(timespec start, timespec end) {
        timespec temp;
        float time_diff = 0;
        if (end.tv_nsec - start.tv_nsec < 0) {
            temp.tv_sec = end.tv_sec - start.tv_sec - 1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        } else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        time_diff = temp.tv_sec * 1000000000.0 + temp.tv_nsec;
        return time_diff / 1000000.0;
    }
 
   unsigned long  diff_time1(timeval start, timeval end) {
        unsigned long time_diff = 0;
        time_diff = (end.tv_sec - start.tv_sec)* 1000000.0 + end.tv_usec - start.tv_usec;
        return time_diff / 1000.0;
    }
    void* thread_func(void *arg) {
	//omp_set_dynamic(0);
    	//omp_set_num_threads(1);
    	mkl_set_num_threads(1);
        unsigned int thread_index = *((unsigned int *)arg);
        fprintf(stderr, "in thread_func created %d thread\n", thread_index);

        // register this thread into predictor
        void* tlr_ptr = g_net.register_thread();
        if (tlr_ptr == NULL) {
            fprintf(stderr, "error register thread\n");
            return NULL;
        }

        std::string line;
        unsigned int i = 0;
        float result[10240];
        size_t len = 0;
        //compute time
        float time = 0;
        unsigned long num = 0;
        for (i = thread_index; i < inputed_lines.size(); i += g_thread_number) {
            line = inputed_lines[i];

            int ret = string_to_input(line, tlr_ptr);
            if (ret == -1){
                fprintf(stderr, 
                    "[ERROR]line %d string to input returned error %d\n", i, ret);
                continue;
            }
            std::vector<float> probs;

            timeval time_beg;// timespec time_beg;
            timeval time_end;// timespec time_end;
            gettimeofday(&time_beg, NULL);// clock_gettime(CLOCK_MONOTONIC, &time_beg);
            ret = g_net.predict(tlr_ptr);
           // clock_gettime(CLOCK_MONOTONIC, &time_end);
           // time += diff_time(time_beg, time_end);
            //time += (end -start) / 1000.0; //CLK_TCK;
            //fprintf(stderr, "thread_num:%d run_time:%f\n", i, time);
             if (ret != 0){
                fprintf(stderr, "[ERROR]predictor returned error %d\n", ret);
                continue;
            }
            gettimeofday(&time_end, NULL);//clock_gettime(CLOCK_MONOTONIC, &time_end);
            if (strcmp(pairwise_demo::blob_name, "") != 0){
                ret = g_net.represent(tlr_ptr, pairwise_demo::blob_name, result, len);
                if (ret != 0){
                    fprintf(stderr, "[ERROR]represent returned error %d\n", ret);
                    continue;
                }
            }
            time += diff_time1(time_beg, time_end);
            num++;
            //end = clock();
            //time += (end -start) / 1000.0; //CLK_TCK;
            //fprintf(stderr, "thread_num:%d run_time:%f\n", i, time);
            probs.resize(1);
            probs[0] = g_net.result_blob(tlr_ptr)->data()[0];
            char buffer[1024];
            *buffer = 0;
    /*        for (size_t j = 0; j < probs.size(); j++){
                snprintf(buffer + strlen(buffer), 1024, "%f", probs[j]);
            }
            if (strcmp(pairwise_demo::blob_name, "") != 0){
                printf("represent:\n");
                for (size_t i = 0; i < len; i++){
                    printf("%f ", result[i]);
                }
                printf("\n");
            }
            std::vector<std::string> vec;
            split2(line, vec, ";");
            std::vector<std::string> vec_info;
            split2(vec[1], vec_info, " ");

            printf("%s\t%s\t%s\n", vec_info[0].c_str(), buffer, vec_info[1].c_str());
*/      
  }
    //   int num = inputed_lines.size()/g_thread_number;
        float avg_time =time / (float) num;
        total_time += avg_time;
        fprintf(stderr, "thread_num_i: %d, input_lines: %d,total_time: %f, run_avg_time: %f\n", thread_index, num, time, avg_time);
        FILE *fp = NULL;
        fp = fopen(time_file, "a");
        fprintf(fp, "thread_num_i: %d, input_lines: %d,total_time: %f, run_avg_time: %f\n", thread_index, num, time, avg_time);
        fclose(fp);
        g_net.destroy_thread(tlr_ptr);
        return NULL;
    }
    void* batch_thread_func(void *arg){
    	mkl_set_num_threads(1);
        unsigned int thread_index = *((unsigned int *)arg);
        fprintf(stderr, "in thread_func created %d thread\n", thread_index);

        // register this thread into predictor
        void* tlr_ptr = g_net.register_thread();
        if (tlr_ptr == NULL){
            fprintf(stderr, "error register thread\n");
            return NULL;
        }

        unsigned int i = 0;
        std::vector<std::string> line_vec;
        for (i = thread_index * g_batch_size; i < inputed_lines.size();
                i += g_thread_number * g_batch_size) {
            line_vec.clear();
            for (size_t j = i; j < g_batch_size + i && j < inputed_lines.size(); j++){
                line_vec.push_back(inputed_lines[j]);
            }
            //printf ("g_batch_size: %d\n", g_batch_size);
            int ret = batch_string_to_input(line_vec, tlr_ptr);
            if (ret == -1){
                fprintf(stderr,
                    "[ERROR]line %d string to input returned error %d\n", i, ret);
                continue;
            }
            ret = g_net.predict(tlr_ptr);
            if (ret != 0){
                fprintf(stderr, "[ERROR]predictor returned error %d\n", ret);
                continue;
            }

            char buffer[1024];
            *buffer = 0;
            //for (int j = 0; j < g_net.result_blob(tlr_ptr)->count(); j++){
            //    printf("line %u thread %d score[%f] ##\n", i + 1 + j, thread_index,
            //        g_net.result_blob(tlr_ptr)->data()[j]);
            //}
        }
        g_net.destroy_thread(tlr_ptr);
        return NULL;
    }

    /*long diff_time(timespec start, timespec end) {
        timespec temp;
        long time_diff = 0;
        if (end.tv_nsec - start.tv_nsec < 0) {
            temp.tv_sec = end.tv_sec - start.tv_sec - 1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        } else {
            temp.tv_sec = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        time_diff = temp.tv_sec * 1000000000 + temp.tv_nsec;
        return time_diff / 1000000000;
    }*/
}

int main(int argc, char *argv[]){
    if (argc < 2){
        printf("Usage:\n\t%s model_name\n", argv[0]);
        printf("\tdemo lego pairwise predictor\n");
        exit(-1);
    }
    if (argc > 2){
        //pairwise_demo::g_thread_number = atoi(argv[4]);
        pairwise_demo::g_thread_number = atoi(argv[2]);
    }
    
    if(0 == strcmp(argv[1], "chinese_ner")) {
        net_proto_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/chinese_ner/for_anakin/nn_net_conf";
        model_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/chinese_ner/for_anakin/epoch20.model";
        input_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/chinese_ner/for_anakin/data_file";
        time_file = "/home/qa_work/CI/workspace/sys_anakin_compare_output/chinese_ner/time/Lego_time.txt";
    } else if (0 == strcmp(argv[1], "language")) {
        net_proto_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/realtitle_lm/liblego.prototxt";
        model_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/realtitle_lm/iter7.model";
        input_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/realtitle_lm/fake_realtitle.txt";
        time_file = "/home/qa_work/CI/workspace/sys_anakin_compare_output/language/time/Lego_time.txt";
    } else if (0 == strcmp(argv[1], "text_classification")) {
        net_proto_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/text_classification/lbs_car_weibo_spot_toutiao_32w_20854/network.define";
        model_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/text_classification/lbs_car_weibo_spot_toutiao_32w_20854/epoch23.model";
        input_file = "/home/qa_work/CI/workspace/baidu/nlp-dnn/liblego/pairwise_demo/model/text_classification/lbs_car_weibo_spot_toutiao_32w_20854/out.ids_lib4.txt";
        time_file = "/home/qa_work/CI/workspace/sys_anakin_compare_output/text_classification/time/Lego_time.txt";
    }

    if (pairwise_demo::g_net.set_net_parameter_from_file(net_proto_file) != 0){
        fprintf(stderr, "error set parameter of net from file : %s", net_proto_file);
        exit(-1);
    }

    if (pairwise_demo::g_net.init(model_file) != 0){
        fprintf(stderr, "error init from file : %s", model_file);
        exit(-1);
    }

    pairwise_demo::load_input_lines(input_file);

    // prepare thread index array, simply numbering from zero
    unsigned int thread_indexes[pairwise_demo::g_thread_number];
    for (unsigned int i = 0; i < pairwise_demo::g_thread_number; i++) {
        thread_indexes[i] = i;
    }
   
    FILE *fp = NULL;
    fp = fopen(time_file, "w");
    fprintf(fp, "clear the outputfile\n");
    fclose(fp);
    pthread_t tids[pairwise_demo::g_thread_number];

    // create each thread and start thread
    timespec time_beg;
    timespec time_end;
    clock_gettime(CLOCK_MONOTONIC, &time_beg);
    if (strcmp(pairwise_demo::blob_name, "") != 0){
        pairwise_demo::g_net.build_represent(pairwise_demo::blob_name);
    }
    for (unsigned int i = 0; i < pairwise_demo::g_thread_number; i++){
        int tmpret = pthread_create(
                            &tids[i], NULL, 
                            pairwise_demo::thread_func,
                            (void*)(&thread_indexes[i]));
        if (tmpret != 0) {
            fprintf(stderr, "pthread_create error, error_code = %d\n", tmpret);
            pairwise_demo::g_net.finalize();
            exit(-1);
        }
        fprintf(stderr, "created %d thread\n", i);
    }
    for (unsigned int i = 0; i < pairwise_demo::g_thread_number; ++i){
        pthread_join(tids[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &time_end);
    float time = pairwise_demo::diff_time(time_beg, time_end);
    //float avg_time = pairwise_demo::total_time / (float)pairwise_demo::g_thread_number;
    // fprintf(stderr, "total_time:%f thread_num:%d  average_time:%f\n",
      //     pairwise_demo::total_time, pairwise_demo::g_thread_number, avg_time);
    fprintf(stderr, "total_time:%f thread_num:%d total_line:%lu average_time:%f QPS: %f\n",
            time, pairwise_demo::g_thread_number, pairwise_demo::inputed_lines.size(),
            (float)time / (float)pairwise_demo::inputed_lines.size(),(float)pairwise_demo::inputed_lines.size()/(float)time*1000.0);

    
    //FILE *fp = NULL;
    fp = fopen(time_file, "a");
    fprintf(fp, "total_time:%f thread_num:%d total_line:%lu average_time:%f QPS: %f\n",
            time, pairwise_demo::g_thread_number, pairwise_demo::inputed_lines.size(),
            (float)time / (float)pairwise_demo::inputed_lines.size(),(float)pairwise_demo::inputed_lines.size()/(float)time*1000.0);
    fclose(fp);
    //std::ofstream fp;
    ////std::string tmp = time_file + "lego_time" + ".txt";
    //char* filename;
    //strcpy(filename, time_file);
    //fp.open(filename);
    //fp << "total_time : " << time << " thread_num : " << pairwise_demo::g_thread_number << "total_line : "<< pairwise_demo::inputed_lines.size() << "avg time : " << (float)time / (float)pairwise_demo::inputed_lines.size() << "QPS : " << (float)pairwise_demo::inputed_lines.size()/(float)time << std::endl;
    //fp.close();

    //compute memory
    liblego::LiblegoMemory liblego_mem;
    pairwise_demo::g_net.compute_memory(liblego_mem);
    printf("layer size: %d\n", liblego_mem.layers_size);
    printf("input blob has %f KB memory\n", (float)liblego_mem.input_blob_memory / 1024);
    size_t total_thread_memory = 0;
    size_t total_model_memory = 0;
    total_thread_memory += liblego_mem.input_blob_memory;
    for (int i = 0; i < liblego_mem.layers_size; i++){
        total_thread_memory += liblego_mem.uniq_blob_memory[i];
        total_thread_memory += liblego_mem.layer_local_buffer_memory[i];
        total_model_memory += liblego_mem.model_param_size[i];

        printf("layer(%d) %s has %lu Bytes model param memory, ",
                i, 
                liblego_mem.layer_type_name[i],
                liblego_mem.model_param_size[i]);
        printf("has %f KB top blobs memory ", (float)liblego_mem.uniq_blob_memory[i] / 1024);
        printf("has %f KB local buffer memory\n",
                (float)liblego_mem.layer_local_buffer_memory[i] / 1024);
    }
    if (total_thread_memory / (1024 * 1024) > 0){
        printf("net need total %f MB thread memory\n", (float)total_thread_memory / (1024 * 1024));
    }
    else{
        printf("net need total %f KB thread memory\n", (float)total_thread_memory / 1024);
    }

    if (total_model_memory / (1024 * 1024) > 0){
        printf("net need total %f MB model memory\n", (float)total_model_memory / (1024 * 1024));
    }
    else{
        printf("net need total %f KB model memory\n", (float)total_model_memory / 1024);
    }
    pairwise_demo::g_net.finalize();
    return 0;
}
