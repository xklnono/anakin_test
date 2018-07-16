
#include "core/context.h"
#include "funcs/func.h"
#include "test_saber_func_power.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncPowerNV, test_func_constructor) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    Context<NV> ctx2;

    NV_API::record_event(event, ctx2.get_compute_stream());

    
    int img_num = 2;
    int in_channels = 3;
    int img_h = 4;
    int img_w = 4;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);
    auto data = img_host.mutable_data();

    for (int i = 0; i < img_host.size(); ++i) {
        data[i] = 0x7f &i;
    }
    img_dev.copy_from(img_host);

    TensorHf4 output_host;
    TensorDf4 output_dev;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);
    
    
    PowerParam<void> param(/*power*/1.2f, /*scale*/ float(1.0/255), /*shift*/0.0f);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Power<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> power;
    power.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    // ! test strategy = SPECIFY
    LOG(INFO) << "|--test strategy = SPECIFY: ";
    power.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    power(input, output, param, ctx1);
    cudaEventSynchronize (event);
    
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);
    
    // ! test query_event function
    NV_API::query_event(event);
 
    // ! test strategy = RUNTIME
    LOG(INFO) << "|--test strategy = RUNTIME: ";
    power.init(input, output, param, RUNTIME, SABER_IMPL, ctx1);
    power(input, output, param, ctx1);
    cudaEventSynchronize (event);
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    // ! test strategy = STATIC 
    LOG(INFO) << "|--test strategy = STATIC: ";
    power.init(input, output, param, STATIC, SABER_IMPL, ctx1);
    power(input, output, param, ctx1);
    cudaEventSynchronize (event);
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    // ! test strategy = UNKNOWN
    LOG(INFO) << "|--test strategy = UNKNOWN: ";
    power.init(input, output, param, UNKNOWN, SABER_IMPL, ctx1);
    power(input, output, param, ctx1);
    cudaEventSynchronize (event);
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncPowerNV, test_func_reset_shape) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    Context<NV> ctx2;

    NV_API::record_event(event, ctx2.get_compute_stream());

    
    int img_num = 2;
    int in_channels = 3;
    int img_h = 4;
    int img_w = 4;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);
    auto data = img_host.mutable_data();

    for (int i = 0; i < img_host.size(); ++i) {
        data[i] = 0x7f &i;
    }
    img_dev.copy_from(img_host);

    TensorHf4 output_host;
    TensorDf4 output_dev;
    TensorDf4 output_dev2;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);
    
    
    PowerParam<void> param(/*power*/1.2f, /*scale*/ float(1.0/255), /*shift*/0.0f);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;
    std::vector<TensorDf4*> output2;

    input.push_back(&img_dev);
    output.push_back(&output_dev);
    output2.push_back(&output_dev2);

    Power<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> power;
    power.compute_output_shape(output, input, param);
    //test reset tensor
    power.reset_output_shape(input, output2, param, ctx1);

    output_dev.re_alloc(output[0]->shape());
    output_dev2.re_alloc(output2[0]->shape());
    output_host.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    power.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    power.init(input, output2, param, SPECIFY, SABER_IMPL, ctx1);

    power(input, output, param, ctx1);
    power(input, output2, param, ctx1);

    cudaEventSynchronize (event);
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    output_host.copy_from(output_dev2);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

