#ifndef ANAKIN_NET_TEST_H
#define ANAKIN_NET_TEST_H

#include <iostream>
#include <fstream>
#include "utils/unit_test/aktest.h"
#include "utils/logger/logger.h"
#include "graph_base.h"
#include "graph.h"
#include "scheduler.h"
#include "net.h"
#include "worker.h"

using namespace anakin;
using ::anakin::test::Test;

using namespace anakin::graph;

/**
 * \brief Graph test is base Test class for anakin graph funciton.  
 */
class NetTest: public Test {
public:
    NetTest(){}

    void SetUp(){}

    void TearDown(){}

protected:
};

#ifdef USE_CUDA
void test_print(Tensor4dPtr<NV, AK_FLOAT>& out_tensor_p) {
    Tensor4d<target_host<NV>::type, AK_FLOAT> h_tensor_result;
    h_tensor_result.re_alloc(out_tensor_p->valid_shape());
    LOG(ERROR) << "result count : " << h_tensor_result.valid_shape().count();
    h_tensor_result.copy_from(*out_tensor_p);
    for (int i = 0; i < h_tensor_result.valid_size(); i++) {
        LOG(INFO) << " GET OUT (" << i << ") " << h_tensor_result.mutable_data()[i];
    }
}

void test_print(Tensor4dPtr<NV, AK_FLOAT>& out_tensor_p, std::ofstream &fp) {
    //Tensor<NV, AK_FLOAT, NCHW> h_tensor_result;
    Tensor4d<target_host<NV>::type, AK_FLOAT> h_tensor_result;
    h_tensor_result.re_alloc(out_tensor_p->valid_shape());
    LOG(INFO) << "num: " << h_tensor_result.num() << " channel: " << h_tensor_result.channel() << " width: " << h_tensor_result.width() << " height: " << h_tensor_result.height();
    LOG(ERROR) << " result count : " << h_tensor_result.valid_size();
    h_tensor_result.copy_from(*out_tensor_p);
    int i = 0;
    for (size_t n = 0; n < h_tensor_result.num(); n++) {
        for(size_t c = 0; c < h_tensor_result.channel(); c++) {
            for(size_t h = 0; h < h_tensor_result.height(); h++) {
                for(size_t w = 0; w < h_tensor_result.width(); w++) {    
                    fp <<  h_tensor_result.mutable_data()[i++] << ", ";
                } 
                fp << std::endl;
            }
            fp << std::endl;
        }
        fp << std::endl;
    }
    fp << "n = " << h_tensor_result.num() << ", ";
    fp << "c = " << h_tensor_result.channel() << ", ";
    fp << "h = " << h_tensor_result.height() << ", ";
    fp << "w = " << h_tensor_result.width() << std::endl;
}
#endif

template<typename Ttype, DataType Dtype>
double tensor_average(Tensor4dPtr<Ttype, Dtype>& out_tensor_p) {
    double sum = 0.0f;
#ifdef USE_CUDA
    float* h_data = new float[out_tensor_p->valid_size()];
    const float* d_data = out_tensor_p->data();
    CUDA_CHECK(cudaMemcpy(h_data, d_data, out_tensor_p->valid_size()*sizeof(float), cudaMemcpyDeviceToHost));
#else
    float* h_data = out_tensor_p->data();
#endif
    for (int i=0; i<out_tensor_p->valid_size(); i++) {
         sum+=h_data[i];
    }
    return sum/out_tensor_p->valid_size();
}

#ifdef USE_X86_PLACE
static int record_dev_tensorfile(const Tensor4d<X86, AK_FLOAT>* dev_tensor, const char* locate) {
    Tensor<target_host<X86>::type, AK_FLOAT, NCHW> host_temp;
    host_temp.re_alloc(dev_tensor->valid_shape());
    host_temp.copy_from(*dev_tensor);
    FILE* fp = fopen(locate, "w+");
    int size = host_temp.valid_shape().count();
    if (fp == 0) {
        LOG(ERROR) << "[ FAILED ] file open target txt: " << locate;
    } else {
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "%.18f \n", i, (host_temp.data()[i]));
        }
        fclose(fp);
    }
    LOG(INFO) << "[ SUCCESS ] Write " << size << " data to: " << locate;
    return 0;
}

#endif

#endif


