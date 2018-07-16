
#include "core/context.h"
#include "funcs/softmax.h"
#include "test_saber_func_softmax_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncSoftmaxNV, test_func_concat_NV) {

    Env<NV>::env_init();
    typedef TargetWrapper<NV> API;

    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 1000;

    int softmax_axis = 3; // channel
    int w_in = 3;
    int h_in = 225;
    int ch_in = 40;
    int num_in = 1;


    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    SoftmaxParam<TensorDf4> param(softmax_axis);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
        ch_in << ", height=" << h_in << ", width=" << w_in;

    LOG(INFO) << "softmax axis= " << param.axis;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);
    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = i % 4;
    }
    TensorDf4 tdin, tdout;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    input_dev_4d.push_back(&tdin);

    // start Reshape & doInfer
    Context<NV> ctx_dev(0, 1, 1);

    Softmax<NV, AK_FLOAT> softmax_dev;

    typedef std::vector<Shape> Shape_v;

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    output_dev_4d.push_back(&tdout);
    softmax_dev.compute_output_shape(output_dev_4d, input_dev_4d, param);

    LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "softmax initialization";

    // ! test strategy = SPECIFY
    LOG(INFO) << "|--test strategy = SPECIFY: ";
    softmax_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    LOG(INFO) << "softmax compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);
    for (int i = 0; i < test_iter; ++i) {
        softmax_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->sync();
    }
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    //print_tensor_device(*output_dev_4d[0]);
    //cudaDeviceSynchronize();

    // ! test strategy = RUNTIME
    LOG(INFO) << "|--test strategy = RUNTIME: ";
    softmax_dev.init(input_dev_4d, output_dev_4d, param, RUNTIME, VENDER_IMPL, ctx_dev);
    softmax_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
    tdout.sync();
    print_tensor_device(tdout);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test strategy = STATIC
    LOG(INFO) << "|--test strategy = STATIC: ";
    softmax_dev.init(input_dev_4d, output_dev_4d, param, STATIC, VENDER_IMPL, ctx_dev);
    softmax_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
    tdout.sync();
    print_tensor_device(tdout);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test strategy = UNKNOWN
    LOG(INFO) << "|--test strategy = UNKNOWN: ";
    softmax_dev.init(input_dev_4d, output_dev_4d, param, UNKNOWN, VENDER_IMPL, ctx_dev);
    softmax_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
    tdout.sync();
    print_tensor_device(tdout);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());

    // ! test reset_output_shape
    LOG(INFO) << "|--test reset_output_shape()";
    TensorDf4 output_dev2;
    std::vector<TensorDf4*> output2;
    output2.push_back(&output_dev2);

    softmax_dev.reset_output_shape(input_dev_4d, output2, param, ctx_dev);
    output_dev2.re_alloc(output2[0]->shape());

    softmax_dev.init(input_dev_4d, output2, param, SPECIFY, VENDER_IMPL, ctx_dev);
    softmax_dev(input_dev_4d, output2, param, ctx_dev);
    output_dev2.sync();
    print_tensor_device(output_dev2);
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

