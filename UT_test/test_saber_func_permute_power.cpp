
#include "core/context.h"
#include "funcs/func.h"
#include "test_saber_func_permute_power.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncPermutePowerNV, test_func_constructor) {

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

    
    //int img_num = 1;
    //int in_channels = 10;
    //int img_h = 480;
    //int img_w = 1440;
    int img_num = 1;
    int in_channels = 3;
    int img_h = 4;
    int img_w = 2;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape img_s(img_num, img_h, img_w, in_channels);

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
    std::vector<int> permute_order= {0, 3, 1, 2};
    PermuteParam<void> permute_param(permute_order);
    //PowerParam<void> power_param(/*power*/1.0f, /*scale*/ float(1.0/255), /*shift*/0.0f);
    PowerParam<void> power_param(/*power*/1.2f, /*scale*/ float(1.0), /*shift*/0.0f);
    
    PermutePowerParam<void> param(permute_param, power_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    PermutePower<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> permute;
    permute.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    // ! test strategy =  SPECITFY
    LOG(INFO) << "|- -test strategy = SPECIFY: " ;
    permute.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    
    permute(input, output, param, ctx1);
    cudaEventSynchronize (event);
    SaberTimer<NV> my_time;
    my_time.start(ctx1);
    for (int i = 0; i < 100; i++) {
        permute(input, output, param, ctx1);
        cudaEventSynchronize (event);
    }
    my_time.end(ctx1);
    LOG(INFO)<<"permute_power cudnn aveage time"<<my_time.get_average_ms()/100;
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
    // ! test strategy =  RUNTIME
    LOG(INFO) << "|- -test strategy = RUNTIME: " ;
    permute.init(input, output, param, RUNTIME, SABER_IMPL, ctx1);
                
    permute(input, output, param, ctx1);
    cudaEventSynchronize (event);
    SaberTimer<NV> my_time1;
    my_time1.start(ctx1);
    for (int i = 0; i < 100; i++) {
        permute(input, output, param, ctx1);
        cudaEventSynchronize (event);
    }   
    my_time1.end(ctx1);
    LOG(INFO)<<"permute_power cudnn aveage time"<<my_time1.get_average_ms()/100;
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
    // ! test strategy =  STATIC
    LOG(INFO) << "|- -test strategy = STATIC: " ;
    permute.init(input, output, param, STATIC, SABER_IMPL, ctx1);

    permute(input, output, param, ctx1);
    cudaEventSynchronize (event);
    SaberTimer<NV> my_time2;
    my_time2.start(ctx1);
    for (int i = 0; i < 100; i++) {
        permute(input, output, param, ctx1);
        cudaEventSynchronize (event);
    }
    my_time2.end(ctx1);
    LOG(INFO)<<"permute_power cudnn aveage time"<<my_time2.get_average_ms()/100;
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
    // ! test strategy =  UNKNOWN
    LOG(INFO) << "|- -test strategy = UNKNOWN: " ;
    permute.init(input, output, param, UNKNOWN, SABER_IMPL, ctx1);

    permute(input, output, param, ctx1);
    cudaEventSynchronize (event);
    SaberTimer<NV> my_time3;
    my_time3.start(ctx1);
    for (int i = 0; i < 100; i++) {
        permute(input, output, param, ctx1);
        cudaEventSynchronize (event);
    }
    my_time3.end(ctx1);
    LOG(INFO)<<"permute_power cudnn aveage time"<<my_time3.get_average_ms()/100;
    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
    print_tensor_host(output_host);

    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncPermutePowerNV, test_func_constructor_reset) {

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

    
    //int img_num = 1;
    //int in_channels = 10;
    //int img_h = 480;
    //int img_w = 1440;
    int img_num = 1;
    int in_channels = 3;
    int img_h = 4;
    int img_w = 2;

    //Shape img_s(img_num, in_channels, img_h, img_w);
    Shape img_s(img_num, img_h, img_w, in_channels);

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
    std::vector<int> permute_order= {0, 3, 1, 2};
    PermuteParam<void> permute_param(permute_order);
    //PowerParam<void> power_param(/*power*/1.0f, /*scale*/ float(1.0/255), /*shift*/0.0f);
    PowerParam<void> power_param(/*power*/1.2f, /*scale*/ float(1.0), /*shift*/0.0f);
    
    PermutePowerParam<void> param(permute_param, power_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;
    std::vector<TensorDf4*> output2;    

    input.push_back(&img_dev);
    output.push_back(&output_dev);
    output2.push_back(&output_dev);

    PermutePower<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> permute;
    permute.compute_output_shape(output, input, param);
    permute.reset_output_shape(input, output2, param, ctx1);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    permute.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    permute.init(input, output2, param, SPECIFY, SABER_IMPL, ctx1);

    permute(input, output, param, ctx1);
    permute(input, output2, param, ctx1);
    cudaEventSynchronize (event);
    SaberTimer<NV> my_time;
    my_time.start(ctx1);
    for (int i = 0; i < 100; i++) {
        permute(input, output2, param, ctx1);
        cudaEventSynchronize (event);
    }
    my_time.end(ctx1);
    LOG(INFO)<<"permute_power cudnn aveage time"<<my_time.get_average_ms()/100;

    output_host.copy_from(output_dev);
    print_tensor_host(img_host);
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

