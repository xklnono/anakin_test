
#include "core/context.h"
#include "funcs/conv_act_pooling.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include "funcs/funcs_param.h"
#include <vector>

using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;


TEST(TestSaberFuncNV, test_func_conv_bn_scale_relu_fusion_reset) {

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 1;
    int kernel_w = 1;
    int g_out_channels = 2;

    int img_num = 1;
    int g_in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    bool bias_term = false;

    LOG(INFO) << " conv param: ";
    LOG(INFO) << " img_num = " << img_num;
    LOG(INFO) << " g_in_channels = " << g_in_channels;
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
    LOG(INFO) << " g_out_channels = " << g_out_channels;

    Shape img_s(img_num, g_in_channels, img_h, img_w);
    Shape weights_s(g_out_channels, g_in_channels, kernel_h, kernel_w);
    Shape bias_s(1, g_out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    TensorDf4 output_dev2;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }
    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO)<<"test conv + bn + scale + relu + pooling ";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param1(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);
    ConvParam<TensorDf4> conv_param(conv_param1);

    ActivationParam<TensorDf4> active_param1(Active_relu);
    ActivationParam<TensorDf4> active_param(active_param);
    typedef typename TensorDf4::Dtype dtype;

    std::vector<dtype> mean, variance;
    std::vector<dtype> scale_w, scale_b;

    mean.resize(g_out_channels);
    variance.resize(g_out_channels);

    for (int i = 0; i < mean.size(); ++i) {
        mean[i] = 0.1f;
    }
    for (int i = 0; i < variance.size(); ++i) {
        variance[i] = 0.1f;
    }

    scale_w.resize(g_out_channels);
    scale_b.resize(g_out_channels);

    for (int i = 0; i < scale_w.size(); ++i) {
        scale_w[i] = 0.1f;
    }
    for (int i = 0; i < scale_b.size(); ++i) {
        scale_b[i] = 0.1f;
    }

    int window_h = 2;
    int window_w = 2;
    int pooling_pad_h = 1;
    int pooling_pad_w = 1;
    int pooling_stride_h = 1;
    int pooling_stride_w = 1;

    BatchnormParam<TensorDf4> batchnorm_param1(mean, variance, 0.5);
    ScaleParam<TensorDf4> scale_param1(scale_w,  scale_b, true);
    BatchnormParam<TensorDf4> batchnorm_param(batchnorm_param1);
    ScaleParam<TensorDf4> scale_param(scale_param1);
    PoolingParam<TensorDf4> pooling_param(window_h, window_w, pooling_pad_h, pooling_pad_w
            , pooling_stride_h, pooling_stride_w, Pooling_max);
    ConvActivePoolingParam<TensorDf4> param1(conv_param,
                    batchnorm_param, scale_param,
                    active_param, pooling_param);
    ConvActivePoolingParam<TensorDf4> param(param1);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;
    std::vector<TensorDf4*> output2;   
 
    input.push_back(&img_dev);
    output.push_back(&output_dev);
    output2.push_back(&output_dev2);

    Conv_act_pooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv;
    conv.compute_output_shape(output, input, param);
    conv.reset_output_shape(input, output2, param, ctx1);
    
    output_dev.re_alloc(output[0]->shape());
    output_dev2.re_alloc(output2[0]->shape());

    conv.init(input, output, param, SPECIFY, SABER_IMPL, ctx1);
    conv.init(input, output2, param, SPECIFY, SABER_IMPL, ctx1);
    conv(input, output, param, ctx1);
    conv(input, output2, param, ctx1);
   
    output_dev2.sync();

    CUDA_CHECK(cudaPeekAtLastError());
}

TEST(TestSaberFuncNV, test_func_conv_bn_scale_relu_fusion) {

    int group = 1;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int kernel_h = 1;
    int kernel_w = 1;
    int g_out_channels = 2;

    int img_num = 1;
    int g_in_channels = 2;
    int img_h = 8;
    int img_w = 8;

    bool bias_term = false;

    LOG(INFO) << " conv param: ";
    LOG(INFO) << " img_num = " << img_num;
    LOG(INFO) << " g_in_channels = " << g_in_channels;
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
    LOG(INFO) << " g_out_channels = " << g_out_channels;

    Shape img_s(img_num, g_in_channels, img_h, img_w);
    Shape weights_s(g_out_channels, g_in_channels, kernel_h, kernel_w);
    Shape bias_s(1, g_out_channels, 1, 1);

    TensorHf4 img_host;
    TensorDf4 img_dev;
    TensorHf4 weights_host;
    TensorDf4 weights_dev;
    TensorHf4 bias_host;
    TensorDf4 bias_dev;
    TensorHf4 output_host;
    TensorDf4 output_dev;
    img_host.re_alloc(img_s);
    img_dev.re_alloc(img_s);

    for (int i = 0; i < img_host.size(); ++i) {
        img_host.mutable_data()[i] = 0x7f & i;
    }
    img_dev.copy_from(img_host);
    weights_host.re_alloc(weights_s);
    weights_dev.re_alloc(weights_s);
    fill_tensor_host_const(weights_host, 1.f);
    weights_dev.copy_from(weights_host);

    if (bias_term) {
        bias_host.re_alloc(bias_s);
        bias_dev.re_alloc(bias_s);
        fill_tensor_host_const(bias_host, 1.f);
        bias_dev.copy_from(bias_host);
    }

    LOG(INFO)<<"test conv + bn + scale + relu + pooling ";

    typedef TargetWrapper<NV> API;
    typename API::event_t event;
    API::create_event(event);

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;

    Context<NV> ctx1(0, 1, 1);
    ConvParam<TensorDf4> conv_param(group, pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    &weights_dev, &bias_dev);

    ActivationParam<TensorDf4> active_param(Active_relu);
    typedef typename TensorDf4::Dtype dtype;

    std::vector<dtype> mean, variance;
    std::vector<dtype> scale_w, scale_b;

    mean.resize(g_out_channels);
    variance.resize(g_out_channels);

    for (int i = 0; i < mean.size(); ++i) {
        mean[i] = 0.1f;
    }
    for (int i = 0; i < variance.size(); ++i) {
        variance[i] = 0.1f;
    }

    scale_w.resize(g_out_channels);
    scale_b.resize(g_out_channels);

    for (int i = 0; i < scale_w.size(); ++i) {
        scale_w[i] = 0.1f;
    }
    for (int i = 0; i < scale_b.size(); ++i) {
        scale_b[i] = 0.1f;
    }

    int window_h = 2;
    int window_w = 2;
    int pooling_pad_h = 1;
    int pooling_pad_w = 1;
    int pooling_stride_h = 1;
    int pooling_stride_w = 1;

    BatchnormParam<TensorDf4> batchnorm_param(mean, variance, 0.5);
    ScaleParam<TensorDf4> scale_param(scale_w,  scale_b, true);

    PoolingParam<TensorDf4> pooling_param(window_h, window_w, pooling_pad_h, pooling_pad_w
            , pooling_stride_h, pooling_stride_w, Pooling_max);
    
    // !test ConvActivePoolingParam <opTensor> &conv_param_in)
    LOG(INFO) << "!--test----- ConvActivePoolingParam <opTensor> &conv_param_in)"; 
    ConvActivePoolingParam<TensorDf4> param4(conv_param,pooling_param);
    //ConvActivePoolingParam<TensorDf4> param(conv_param,
    //                batchnorm_param, scale_param,
    //                active_param, pooling_param);

    std::vector<TensorDf4*> input;
    std::vector<TensorDf4*> output;

    input.push_back(&img_dev);
    output.push_back(&output_dev);

    Conv_act_pooling<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW> conv;
    conv.compute_output_shape(output, input, param4);

    output_dev.re_alloc(output[0]->shape());
    // ! test stategy = SPECIFY
    LOG(INFO) << "|--test stategy = SPECIFY:";
    conv.init(input, output, param4, SPECIFY, SABER_IMPL, ctx1);

    conv(input, output, param4, ctx1);
    output_dev.sync();

    CUDA_CHECK(cudaPeekAtLastError());
    // !test ConvActivePoolingParam <opTensor> &conv_param_in, activation)
    LOG(INFO) << "!--test ConvActivePoolingParam <opTensor> &conv_param_in ,activation";
    ConvActivePoolingParam<TensorDf4> param3(conv_param,active_param);
    conv.compute_output_shape(output, input, param4);

    output_dev.re_alloc(output[0]->shape());
    LOG(INFO) << "|--test stategy = SPECIFY:";
    conv.init(input, output, param3, SPECIFY, SABER_IMPL, ctx1);

    conv(input, output, param3, ctx1);
    output_dev.sync();

    CUDA_CHECK(cudaPeekAtLastError());
    // !test ConvActivePoolingParam <opTensor> &conv_param_in, pooling)
    input.push_back(&img_dev);
    output.push_back(&output_dev);

    LOG(INFO) << "!--test ConvActivePoolingParam <opTensor> &conv_param_in ,pooling";
    ConvActivePoolingParam<TensorDf4> param2(conv_param);
    conv.compute_output_shape(output, input, param2);
    
    output_dev.re_alloc(output[0]->shape());
    LOG(INFO) << "|--test stategy = SPECIFY:";
    conv.init(input, output, param2, SPECIFY, SABER_IMPL, ctx1);

    conv(input, output, param2, ctx1);
    output_dev.sync();

    CUDA_CHECK(cudaPeekAtLastError());
    // !test ConvActivePoolingParam <opTensor> &conv_param_in,activation, pooling)
    LOG(INFO) << "!--test ConvActivePoolingParam <opTensor> &conv_param_in ,activation,pooling";
    ConvActivePoolingParam<TensorDf4> param1(conv_param, active_param, pooling_param);
    conv.compute_output_shape(output, input, param1);

    output_dev.re_alloc(output[0]->shape());
    LOG(INFO) << "|--test stategy = SPECIFY:";
    conv.init(input, output, param1, SPECIFY, SABER_IMPL, ctx1);

    conv(input, output, param1, ctx1);
    output_dev.sync();

    CUDA_CHECK(cudaPeekAtLastError());
    ConvActivePoolingParam<TensorDf4> param(conv_param,
                        batchnorm_param, scale_param,
                            active_param, pooling_param);

    // ! test stategy = RUNTIME
    LOG(INFO) << "|--test stategy = RUNTIME:";
    conv.init(input, output, param, RUNTIME, SABER_IMPL, ctx1);

    conv(input, output, param, ctx1);
    output_dev.sync();

    CUDA_CHECK(cudaPeekAtLastError());
    // ! test stategy = STATIC
    LOG(INFO) << "|--test stategy = STATIC:";
    conv.init(input, output, param, STATIC, SABER_IMPL, ctx1);

    conv(input, output, param, ctx1);
    output_dev.sync();

    CUDA_CHECK(cudaPeekAtLastError());

}


int main(int argc, const char** argv){

    Env<NV>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

