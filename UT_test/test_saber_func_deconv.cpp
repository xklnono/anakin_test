
#include "core/context.h"
#include "funcs/deconv.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/funcs_param.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncNV, test_func_deconv) {

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

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 2;
    int dilation_w = 2;

    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 16;
    
    int img_num = 1;
    int in_channels = 16;
    int img_h = 256;
    int img_w = 256;

    bool bias_term = true;

    LOG(INFO)<<"conv param: ";
    LOG(INFO)<<" img_num = "<<img_num;
    LOG(INFO)<<" in_channels = "<<in_channels;
    LOG(INFO)<<" img_h = "<<img_h;
    LOG(INFO)<<" img_w = "<<img_w;
    LOG(INFO)<<" group = "<<group;
    LOG(INFO)<<" pad_h = "<<pad_h;
    LOG(INFO)<<" pad_w = "<<pad_w;
    LOG(INFO)<<" stride_h = "<<stride_h;
    LOG(INFO)<<" stride_w = "<<stride_w;
    LOG(INFO)<<" dilation_h = "<<dilation_h;
    LOG(INFO)<<" dilation_w = "<<dilation_w;
    LOG(INFO)<<" kernel_h = "<<kernel_h;
    LOG(INFO)<<" kernel_w = "<<kernel_w;
    LOG(INFO)<<" out_channels = "<<out_channels;

    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape weights_s(out_channels, in_channels, kernel_h, kernel_w);
    Shape bias_s(1, out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f &i;
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
    TensorDf4 output_dev2;

    // start Reshape & doInfer

    Context<NV> ctx1(0, 1, 1);
    
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               &weights_dev, &bias_dev);
    ActivationParam<TensorDf4> active_param(Active_relu);
    
    ConvActiveParam<TensorDf4> param1(conv_param);
    ConvActiveParam<TensorDf4> param(conv_param, active_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;
    // ! test reset_output_shape
    std::vector<TensorDf4*> output2;
    
    input.push_back(&img_dev);
    output.push_back(&output_dev);
    output2.push_back(&output_dev2);

    Deconv<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv;
    conv.compute_output_shape(output, input, param);
    conv.reset_output_shape(input, output2, param, ctx1);

    output_dev.re_alloc(output[0]->shape());
    output_host.re_alloc(output[0]->shape());
    output_dev2.re_alloc(output2[0]->shape());

    LOG(INFO)<<"regular start with group = "<<group;
    // init assume output tensor has been reshpaed by user.
    // ! test strategy = RUNTIME
    LOG(INFO) << "|-- test stategy = RUNTIME: ";
    conv.init(input, output, param, RUNTIME, VENDER_IMPL, ctx1);
    conv.init(input, output2, param, RUNTIME, VENDER_IMPL, ctx1); 
   
    conv(input, output, param, ctx1);
    output_host.copy_from(output_dev);
    conv(input, output2, param, ctx1);
    output_host.copy_from(output_dev2);

   // conv.compute_output_shape(output, input, param);

    param.conv_param.group = 1;
    param.conv_param.pad_h = 1;
    param.conv_param.pad_w = 1;
    LOG(INFO)<<" param changed start with group = "<<param.conv_param.group;
   // conv(input, output, param, ctx1);

   // output_host.reshape(output[0]->shape());
   // output_host.copy_from(output_dev);

    CUDA_CHECK(cudaPeekAtLastError());
    // ! test strategy = SPECIFY
    LOG(INFO) << "|-- test stategy = SPECIFY: ";
    conv.init(input, output, param, SPECIFY, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);
    output_host.copy_from(output_dev);

    conv.compute_output_shape(output, input, param);

    param.conv_param.group = 1;
    param.conv_param.pad_h = 1;
    param.conv_param.pad_w = 1;
    LOG(INFO)<<" param changed start with group = "<<param.conv_param.group;
    conv(input, output, param, ctx1);

    output_host.reshape(output[0]->shape());
    output_host.copy_from(output_dev);

    CUDA_CHECK(cudaPeekAtLastError());
    // ! test strategy = STATIC
    LOG(INFO) << "|-- test stategy = STATIC: ";
    conv.init(input, output, param, STATIC, VENDER_IMPL, ctx1);

    conv(input, output, param, ctx1);
    output_host.copy_from(output_dev);

    conv.compute_output_shape(output, input, param);

    param.conv_param.group = 1;
    param.conv_param.pad_h = 1;
    param.conv_param.pad_w = 1;
    LOG(INFO)<<" param changed start with group = "<<param.conv_param.group;
    conv(input, output, param, ctx1);

    output_host.reshape(output[0]->shape());
    output_host.copy_from(output_dev);

    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, const char** argv){
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

