
#include "core/context.h"
#include "funcs/pooling.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/funcs_param.h"
#include "saber_types.h"
#include "funcs/timer.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncNV, test_func_pooling) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 4;
    int img_h = 800;
    int img_w = 1440;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }
    img_dev.copy_from(img_host);
    
    TensorHf4 output_host;
    TensorDf4 output_dev;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);
    int window_h = 2;
    int window_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    LOG(INFO)<<" img_num: " << img_num;
    LOG(INFO)<<" in_channels: " << in_channels;
    LOG(INFO)<<" img_h: " << img_h;
    LOG(INFO)<<" img_w: " << img_w;
    LOG(INFO)<<" window_h: " << window_h;
    LOG(INFO)<<" window_w: " << window_w;
    LOG(INFO)<<" pad_h: " << pad_h;
    LOG(INFO)<<" pad_w: " << pad_w;
    LOG(INFO)<<" stride_h: " << stride_h;
    LOG(INFO)<<" stride_w: " << stride_w;

    PoolingParam<TensorDf4> param(window_h, window_w, pad_h, pad_w
            , stride_h, stride_w, Pooling_max);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Pooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> pooling;
    pooling.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    pooling.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling(input, output, param, ctx1);

    SaberTimer<NV> t1;
    int ts = 1000;

    for (int i = 0; i < ts; ++i) {
        t1.start(ctx1);
        pooling(input, output, param, ctx1);
        output[0]->sync();
        t1.end(ctx1);
    }
    output_dev.sync();
    cudaDeviceSynchronize();
    LOG(INFO)<<" average time: "<<t1.get_average_ms()<<" ms";
    LOG(INFO)<<" tile 10% time: "<<t1.get_tile_time(10)<<" ms";
    LOG(INFO)<<" tile 50% time: "<<t1.get_tile_time(50)<<" ms";
    LOG(INFO)<<" tile 90% time: "<<t1.get_tile_time(90)<<" ms";
    LOG(INFO)<<" tile 95% time: "<<t1.get_tile_time(95)<<" ms";
    LOG(INFO)<<" tile 99% time: "<<t1.get_tile_time(99)<<" ms";

    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_pooling_result) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }
    img_dev.copy_from(img_host);

    TensorDf4 output_dev;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);
    int window_h = 2;
    int window_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;

            LOG(INFO)<<" img_num: " << img_num;
            LOG(INFO)<<" in_channels: " << in_channels;
            LOG(INFO)<<" img_h: " << img_h;
            LOG(INFO)<<" img_w: " << img_w;
            LOG(INFO)<<" window_h: " << window_h;
            LOG(INFO)<<" window_w: " << window_w;
            LOG(INFO)<<" pad_h: " << pad_h;
            LOG(INFO)<<" pad_w: " << pad_w;
            LOG(INFO)<<" stride_h: " << stride_h;
            LOG(INFO)<<" stride_w: " << stride_w;

    PoolingParam<TensorDf4> param(window_h, window_w, pad_h, pad_w
            , stride_h, stride_w, Pooling_max);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Pooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> pooling;
    pooling.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());

    // init assume output tensor has been reshpaed by user.
    // ! test strategy = SPECIFY
    LOG(INFO) << "|--test strategy = SPECIFY: ";
    pooling.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling(input, output, param, ctx1);

    output_dev.sync();
    print_tensor_device(output_dev);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test strategy = RUNTIME
    LOG(INFO) << "|--test strategy = RUNTIME: ";
    pooling.init(input, output, param, RUNTIME, VENDER_IMPL, ctx1);
    pooling(input, output, param, ctx1);
    output_dev.sync();
    print_tensor_device(output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test strategy = STATIC
    LOG(INFO) << "|--test strategy = STATIC: ";
    pooling.init(input, output, param, STATIC, VENDER_IMPL, ctx1);
    pooling(input, output, param, ctx1);
    output_dev.sync();
    print_tensor_device(output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test strategy = UNKNOWN
    LOG(INFO) << "|--test strategy = UNKNOWN: ";
    pooling.init(input, output, param, UNKNOWN, VENDER_IMPL, ctx1);
    pooling(input, output, param, ctx1);
    output_dev.sync();
    print_tensor_device(output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test reset_output_shape
    LOG(INFO) << "|--test reset_output_shape()";
    TensorDf4 output_dev2;
    std::vector<TensorDf4*> output2;
    output2.push_back(&output_dev2);

	pooling.reset_output_shape(input, output2, param, ctx1);
    output_dev2.re_alloc(output2[0]->shape());

    pooling.init(input, output2, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling(input, output2, param, ctx1);
    output_dev2.sync();
    print_tensor_device(output_dev2);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test add branch coverage in compute_output_shape
    LOG(INFO) << "|--test add param init use pooling_unknow";
    bool pooling_in = true;
    PoolingParam<TensorDf4> param_pooling_unknow(window_h, window_w, pad_h, pad_w
            , stride_h, stride_w, Pooling_max, pooling_in);
    pooling.init(input, output, param_pooling_unknow, SPECIFY, VENDER_IMPL, ctx1);
    pooling(input, output, param_pooling_unknow, ctx1);
    output_dev.sync();
    print_tensor_device(output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
    
    // ! test add branch coverage in compute_output_shape
    pad_h = 0;
    LOG(INFO) << "|--test param in param.pooling_padded()==false";
    PoolingParam<TensorDf4> param_pad_h_0(window_h, window_w, pad_h, pad_w
            , stride_h, stride_w, Pooling_max);
    pooling.init(input, output, param_pad_h_0, SPECIFY, VENDER_IMPL, ctx1);
    pooling(input, output, param_pad_h_0, ctx1);
    output_dev.sync();
    print_tensor_device(output_dev);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_pooling_shared_buffer) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    int img_num = 1;
    int in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    Shape img_s(img_num, in_channels, img_h, img_w);

    TensorHf4 img_host;
    TensorDf4 img_dev;

    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }
    img_dev.copy_from(img_host);

    TensorDf4 t0;
    TensorDf4 t1;
    Shape img_s_sub(img_num, in_channels, img_h/2, img_w/2);

    t0.share_sub_buffer(img_dev, img_s_sub, {0,0,0,0});
    t1.share_sub_buffer(img_dev, img_s_sub, {0,0,4,4});

    TensorDf4 output_dev;

    TensorDf4 out0;
    TensorDf4 out1;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);
    int window_h = 2;
    int window_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;

    LOG(INFO)<<" img_num: " << img_num;
    LOG(INFO)<<" in_channels: " << in_channels;
    LOG(INFO)<<" img_h: " << img_h;
    LOG(INFO)<<" img_w: " << img_w;
    LOG(INFO)<<" window_h: " << window_h;
    LOG(INFO)<<" window_w: " << window_w;
    LOG(INFO)<<" pad_h: " << pad_h;
    LOG(INFO)<<" pad_w: " << pad_w;
    LOG(INFO)<<" stride_h: " << stride_h;
    LOG(INFO)<<" stride_w: " << stride_w;

    PoolingParam<TensorDf4> param(window_h, window_w, pad_h, pad_w
            , stride_h, stride_w, Pooling_max);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Pooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> pooling;
    Pooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> pooling0;
    Pooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> pooling1;

    pooling.compute_output_shape(output, input, param);

    Shape total_shape = output[0]->shape();

    output_dev.re_alloc(total_shape);
    Shape out_sub_shape = {total_shape[0], total_shape[1], total_shape[2]/2, total_shape[3]/2};

    out0.share_sub_buffer(output_dev, out_sub_shape, {0,0,0,0});
    out1.share_sub_buffer(output_dev, out_sub_shape, {0,0,out_sub_shape[2], out_sub_shape[3]});

    std::vector<TensorDf4*> input0, input1;
    std::vector<TensorDf4*> output0, output1;

    input0.push_back(&t0);
    input1.push_back(&t1);
    output0.push_back(&out0);
    output1.push_back(&out1);

    // init assume output tensor has been reshpaed by user.
    pooling0.init(input0, output0, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling0(input0, output0, param, ctx1);

    pooling1.init(input1, output1, param, SPECIFY, VENDER_IMPL, ctx1);
    pooling1(input1, output1, param, ctx1);

    out0.sync();
    out1.sync();

    print_tensor_device(output_dev);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

