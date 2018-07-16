
#include "core/context.h"
#include "funcs/conv.h"
#include "funcs/reshape.h"
#include "funcs/timer.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/funcs_param.h"
#include "saber_types.h"
#include <vector>



using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

//! test: use conv node to test timer.cpp in NV
TEST(TestSaberFuncNV, test_func_timer_NV) {

    int group = 2;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 2;
    
    int img_num = 1;
    int in_channels = 2;
    int img_h = 188;
    int img_w = 488;

    bool bias_term = true;

    LOG(INFO) << "conv param: ";
    LOG(INFO) << " img_num = " << img_num;
    LOG(INFO) << " in_channels = " << in_channels;
    LOG(INFO) << " img_h = " << img_h;
    LOG(INFO) << " img_w = " << img_w;
    LOG(INFO) << " group = " << group;
    LOG(INFO) << " pad_h = " << pad_h;
    LOG(INFO) << " pad_w = " << pad_w;
    LOG(INFO) << " stride_h = " << stride_h;
    LOG(INFO) << " stride_w = " << stride_w;
    LOG(INFO) << " dilation_h = " << dilation_h;
    LOG(INFO) << " dilation_w = " << dilation_w;
    LOG(INFO) << " kernel_h = " << kernel_h;
    LOG(INFO) << " kernel_w = " << kernel_w;
    LOG(INFO) << " out_channels = " << out_channels;

    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape bias_s(1, out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 63 & i;
    }

    img_dev.copy_from(img_host);
    
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);

    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    TensorHf4 bias_host;
    TensorDf4 bias_dev;

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);

        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    TensorHf4 output_host;
    TensorDf4 output_dev;

    // start Reshape & doInfer
    Context<NV> ctx1(0, 1, 1);
    
    ConvParam<TensorDf4> param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv;
    conv.compute_output_shape(output, input, param);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());

    LOG(INFO) << "regular start with group = " << group;
    // init assume output tensor has been reshpaed by user.
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    SaberTimer<NV> t1;
    t1.clear();

    //! test get_average_ms() when ms_time.size == 0 
    LOG(INFO)<<" average time when ms_time.size==0: "<<t1.get_average_ms()<<" ms";
    //! test get_tile_time() when ms_time.size == 0 
    LOG(INFO)<<" tile 10% time when ms_time.size==0: "<<t1.get_tile_time(10)<<" ms";

    int ts = 100;
    for (int i = 0; i < ts; ++i) {
        t1.start(ctx1);
        conv(input, output, param, ctx1);
        output[0]->sync();
        t1.end(ctx1);
    }
    output_dev.sync();
    print_tensor_device(output_dev);

    cudaDeviceSynchronize();

    //! test get_average_ms() 
    LOG(INFO)<<" average time: "<<t1.get_average_ms()<<" ms";
    LOG(INFO)<<" tile 10% time: "<<t1.get_tile_time(10)<<" ms";
    LOG(INFO)<<" tile 50% time: "<<t1.get_tile_time(50)<<" ms";
    LOG(INFO)<<" tile 90% time: "<<t1.get_tile_time(90)<<" ms";
    LOG(INFO)<<" tile 95% time: "<<t1.get_tile_time(95)<<" ms";
    LOG(INFO)<<" tile 99% time: "<<t1.get_tile_time(99)<<" ms";

    //! test get_tile_time() when tile <0 || tile > 100
    LOG(INFO)<<" tile -1% time: "<<t1.get_tile_time(-1)<<" ms";
    LOG(INFO)<<" tile 108% time: "<<t1.get_tile_time(108)<<" ms";

    //test timer.get_time_stat() func
    std::list<float> ms_time;
    ms_time = t1.get_time_stat();
    
    int total_items = ms_time.size();
    auto it = ms_time.begin();
    for (int i = 0; i < total_items; ++i) {
        LOG(INFO)<<"|--TIME RECORD IN SaberTimer List, item["<<i<<"]"<<*it;
        ++it;
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

//! test: use conv node to test timer.cpp in X86
TEST(TestSaberFuncNV, test_func_timer_X86) {

    Env<NV>::env_init();
    Env<X86>::env_init();
    typedef TargetWrapper<NV> API;
    typedef TargetWrapper<X86> X86_API;

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<X86, AK_FLOAT, HW> TensorHf2;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<NV, AK_FLOAT, HW> TensorDf2;

    typedef TensorHf2::Dtype dtype;

    int w_in = 8;
    int h_in = 8;
    int ch_in = 44;
    int num_in = 22;

    std::vector<int> shape_param_4d = {0, 0, -1, 16};

    ReshapeParam<TensorHf4> param_host_4d(shape_param_4d);

    LOG(INFO) << "Reshape param: ";
    LOG(INFO) << " input size, num=" << num_in << ", channel=" << \
        ch_in << ", height=" << h_in << ", width=" << w_in;
    LOG(INFO) << "4d reshape params = " << shape_param_4d[0] << ", " \
        << shape_param_4d[1] << ", " << shape_param_4d[2] << \
              ", " << shape_param_4d[3];

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out_4d(num_in, ch_in, 4, 16);

    TensorHf4 thost_in, thost_out_4d;

    thost_in.re_alloc(shape_in);

    for (int i = 0; i < thost_in.size(); ++i){
        thost_in.mutable_data()[i] = static_cast<dtype>(i);
    }

	// start Reshape & doInfer
    Context<X86> ctx_host;

    std::vector<TensorHf4*> input_host_4d;
    std::vector<TensorHf4*> output_host_4d;

    input_host_4d.push_back(&thost_in);

    Reshape<X86, AK_FLOAT> host_reshape_4d;
    Reshape<NV, AK_FLOAT> dev_reshape_4d;
    Reshape<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, HW, NCHW, HW> host_reshape_2d;
    Reshape<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, HW, NCHW, HW> dev_reshape_2d;

    typedef std::vector<Shape> Shape_v;

    output_host_4d.push_back(&thost_out_4d);

    LOG(INFO) << "reshape compute output shape";
    host_reshape_4d.compute_output_shape(output_host_4d, input_host_4d, param_host_4d);

    LOG(INFO) << "shape out 4d: " << shape_out_4d[0] << ", " << shape_out_4d[1] << ", " << \
              shape_out_4d[2] << ", " << shape_out_4d[3];

    thost_out_4d.re_alloc(shape_out_4d);

    // init assume output tensor has been reshpaed by user.
    LOG(INFO) << "reshape initialization";
    host_reshape_4d.init(input_host_4d, output_host_4d, param_host_4d, \
    	SPECIFY, VENDER_IMPL, ctx_host);


    SaberTimer<X86> t2;
    t2.clear();
    //! test get_average_ms() when ms_time.size == 0
    LOG(INFO)<<" average time when ms_time.size==0: "<<t2.get_average_ms()<<" ms";
    //! test get_tile_time() when ms_time.size == 0
    LOG(INFO)<<" tile 10% time when ms_time.size==0: "<<t2.get_tile_time(10)<<" ms";
    
    int ts = 100;
    for (int i = 0; i < ts; ++i) {
        t2.start(ctx_host);
        host_reshape_4d(input_host_4d, output_host_4d, param_host_4d, ctx_host);
        output_host_4d[0]->sync();
        t2.end(ctx_host);
    }
    print_tensor_host(thost_out_4d);
    LOG(INFO)<<" average time: "<<t2.get_average_ms()<<" ms";
    LOG(INFO)<<" tile 10% time: "<<t2.get_tile_time(10)<<" ms";
    LOG(INFO)<<" tile 50% time: "<<t2.get_tile_time(50)<<" ms";
    LOG(INFO)<<" tile 90% time: "<<t2.get_tile_time(90)<<" ms";
    LOG(INFO)<<" tile 95% time: "<<t2.get_tile_time(95)<<" ms";
    LOG(INFO)<<" tile 99% time: "<<t2.get_tile_time(99)<<" ms";

    //! test get_tile_time() when tile <0 || tile > 100
    LOG(INFO)<<" tile -1% time: "<<t2.get_tile_time(-1)<<" ms";
    LOG(INFO)<<" tile 108% time: "<<t2.get_tile_time(108)<<" ms";

    //test timer.get_time_stat() func
    std::list<float> ms_time;
    ms_time = t2.get_time_stat();

    int total_items = ms_time.size();
    auto it = ms_time.begin();
    for (int i = 0; i < total_items; ++i) {
        LOG(INFO)<<"|--TIME RECORD IN SaberTimer List, item["<<i<<"]"<<*it;
        ++it;
    }

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}


int main(int argc, const char** argv){
    anakin::saber::Env<NV>::env_init();

    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

