#include "test_saber_tensor_NVH.h"
#include "tensor_op.h"
#include <vector>
#include "core/target_traits.h"
#include "core/env.h"

using namespace anakin::saber;

typedef TargetWrapper<NVHX86> NVH_API;
typedef TargetWrapper<NV> NV_API;
typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
typedef Tensor<X86, AK_FLOAT, HW> TensorHf2;
typedef Tensor<NV, AK_FLOAT, HW> TensorDf2;
typedef TensorHf4::Dtype dtype;

TEST(TestSaberTensorNV, test_tensor_constructor) {

    //! test empty constructor
    LOG(INFO) << "test default (empty) constructor";
    TensorHf4 thost0;
    TensorDf4 tdev0;

    //! test tensor re_alloc function empty constructor
    Shape sh0(2, 3, 10, 10);
    LOG(INFO) << "|--test tensor re_alloc function on empty tensor";
    thost0.re_alloc(sh0);
    tdev0.re_alloc(sh0);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    LOG(INFO) << "|--tensor size of device: " << tdev0.size();

    //*************NEW******************
    //! test tensor count()
    LOG(INFO) << "|--thost count(0,0) of host: " << thost0.count(0,0);
    CHECK_EQ(thost0.count(0,0), 1) << "error with host tensor count(0,0)";
    LOG(INFO) << "|--thost count(0,0) of device: " << tdev0.count(0,0);
    CHECK_EQ(tdev0.count(0,0), 1) << "error with dev tensor count(0,0)";

    LOG(INFO) << "|--thost count(0,1) of host: " << thost0.count(0,1);
    CHECK_EQ(thost0.count(0,1), 2) << "error with host tensor count(0,1)";
    LOG(INFO) << "|--thost count(0,1) of device: " << tdev0.count(0,1);
    CHECK_EQ(tdev0.count(0,1), 2) << "error with dev tensor count(0,1)";

    LOG(INFO) << "|--thost count(0,2) of host: " << thost0.count(0,2);
    CHECK_EQ(thost0.count(0,2), 6) << "error with host tensor count(0,2)";
    LOG(INFO) << "|--thost count(0,2) of device: " << tdev0.count(0,2);
    CHECK_EQ(tdev0.count(0,2), 6) << "error with dev tensor count(0,2)";

    LOG(INFO) << "|--thost count(0,3) of host: " << thost0.count(0,3);
    CHECK_EQ(thost0.count(0,3), 60) << "error with host tensor count(0,3)";
    LOG(INFO) << "|--thost count(0,3) of device: " << tdev0.count(0,3);
    CHECK_EQ(tdev0.count(0,3), 60) << "error with dev tensor count(0,3)";

    LOG(INFO) << "|--thost count(0,4) of host: " << thost0.count(0,4);
    CHECK_EQ(thost0.count(0,4), 600) << "error with host tensor count(0,4)";
    LOG(INFO) << "|--thost count(0,4) of device: " << tdev0.count(0,4);
    CHECK_EQ(tdev0.count(0,4), 600) << "error with dev tensor count(0,4)";

    //TODO:abnormal case in func:count() 
    //CHECK_EQ(thost0.count(-1,4), 0) << "error with host tensor count(-1,4)";
    //CHECK_EQ(thost0.count(3,5), 0) << "error with host tensor count(3,5)";
    //CHECK_EQ(thost0.count(3,1), 0) << "error with host tensor count(3,5)";

    //! test shape dims(),count(),size()
    LOG(INFO) << "|--shape0 dims of shape(2,3,10,10): " << sh0.dims();
    CHECK_EQ(sh0.dims(), 4) << "error with sh0 dims of shape(2,3,10,10)";
    LOG(INFO) << "|--shape0 count of shape(2,3,4,5):" << sh0.count();
    CHECK_EQ(sh0.count(), 600) << "error with sh0 count of shape(2,3,10,10)";
    LOG(INFO) << "|--sh0.count(1):" << sh0.count(1);
    CHECK_EQ(sh0.count(1), 300) << "error with sh0 count of shape(2,3,10,10)";
    LOG(INFO) << "|--sh0.count(2):" << sh0.count(2);
    CHECK_EQ(sh0.count(2), 100) << "error with sh0 count of shape(2,3,10,10)";
    LOG(INFO) << "|--sh0.count(3):" << sh0.count(3);
    CHECK_EQ(sh0.count(3), 10) << "error with sh0 count of shape(2,3,10,10)";
    LOG(INFO) << "|--sh0.count(4):" << sh0.count(4);
    CHECK_EQ(sh0.count(4), 1) << "error with sh0 count of shape(2,3,10,10)";
    LOG(INFO) << "|--shape0 size of shape(2,3,4,5):" << sh0.size();
    CHECK_EQ(sh0.size(), 4) << "error with sh0 size of shape(2,3,,10)";

    //! test tensor host dims(),size(),valid_size()
    Shape shtest1(1, 2, 3);
    LOG(INFO) << "|--thost0.dims():" << thost0.dims();
    CHECK_EQ(thost0.dims(), 4) << "error with thost0 dims()";
    LOG(INFO) << "|--thost0.size():" << thost0.size();
    CHECK_EQ(thost0.size(), 600) << "error with thost0 size()";
    LOG(INFO) << "|--thost0.valid_size():" << thost0.valid_size();
    CHECK_EQ(thost0.valid_size(), 600) << "error with thost0 valid_size()";

    //! test tensor device dims(),size(),valid_size()
    LOG(INFO) << "|--tdev0.dims():" << tdev0.dims();
    CHECK_EQ(tdev0.dims(), 4) << "error with tdev0 dims()";
    LOG(INFO) << "|--tdev0.size():" << tdev0.size();
    CHECK_EQ(tdev0.size(), 600) << "error with tdev0 size()";
    LOG(INFO) << "|--tdev0.valid_size():" << tdev0.valid_size();
    CHECK_EQ(tdev0.valid_size(), 600) << "error with tdev0 valid_size()";

    //! test tensor dev n/c/h/w and n/c/h/w_index()
    LOG(INFO) << "|--tdev0.num():" << tdev0.num();
    CHECK_EQ(tdev0.num(), 2) << "error with tdev0 num()";
    LOG(INFO) << "|--tdev0.num_index():" << tdev0.num_index();
    CHECK_EQ(tdev0.num_index(), 0) << "error with tdev0 num_index()";
    LOG(INFO) << "|--tdev0.channel():" << tdev0.channel();
    CHECK_EQ(tdev0.channel(), 3) << "error with tdev0 channel()";
    LOG(INFO) << "|--tdev0.channel_index():" << tdev0.channel_index();
    CHECK_EQ(tdev0.channel_index(), 1) << "error with tdev0 channel_index()";
    LOG(INFO) << "|--tdev0.height():" << tdev0.height();
    CHECK_EQ(tdev0.height(), 10) << "error with tdev0 height()";
    LOG(INFO) << "|--tdev0.height_index():" << tdev0.height_index();
    CHECK_EQ(tdev0.height_index(), 2) << "error with tdev0 height_index()";
    LOG(INFO) << "|--tdev0.width():" << tdev0.width();
    CHECK_EQ(tdev0.width(), 10) << "error with tdev0 width()";
    LOG(INFO) << "|--tdev0.width_index():" << tdev0.width_index();
    CHECK_EQ(tdev0.width_index(), 3) << "error with tdev0 width_index()";
    LOG(INFO) << "|--tdev0.device_id():" << tdev0.device_id();
    CHECK_EQ(tdev0.device_id(), 0) << "error with tdev0 device_id()";

    fill_tensor_device_const(tdev0, 1.44f);

    //! test shape is_continue()
    //is_continue is shp1 ?= shp2
    Shape shtest2;
    shtest2 = tdev0.shape();
    Shape shtemp(1, 3, 10, 10);
    LOG(INFO) << "|--shtest2.is_continue(shtemp):" << shtest2.is_continue(shtemp);
    CHECK_EQ(shtest2.is_continue(shtemp), 1) << "error with shtest2.is_continue(shtemp)";
    LOG(INFO) << "|--shtest2.is_continue(sh0):" << shtest2.is_continue(sh0);
    CHECK_EQ(shtest2.is_continue(sh0), 1) << "error with shtest2.is_continue(sh0)";
    LOG(INFO) << "|--shtest2.is_continue(shtest1):" << shtest2.is_continue(shtest1);
    CHECK_EQ(shtest2.is_continue(shtest1), 0) << "error with shtest2.is_continue(shtest1)";

    //! test Shape zero() and minusone()
    Shape shtest3(44, 55, 66, 77);
    Shape shtest4;
    LOG(INFO) << "|--shape of tensor: " << shtest3[0] << ", " << shtest3[1] << ", " << shtest3[2] << ", " <<shtest3[3];
    CHECK_EQ(shtest3[0], 44) << "error with shtest3[0]";
    CHECK_EQ(shtest3[1], 55) << "error with shtest3[1]";
    CHECK_EQ(shtest3[2], 66) << "error with shtest3[2]";
    CHECK_EQ(shtest3[3], 77) << "error with shtest3[3]";

    shtest4 = Shape::zero(4);
    LOG(INFO) << "|--shape of tensor: " << shtest4[0] << ", " << shtest4[1] << ", " << shtest4[2] << ", " <<shtest4[3];
    CHECK_EQ(shtest4[0], 0) << "error with shtest4[0]";
    CHECK_EQ(shtest4[1], 0) << "error with shtest4[1]";
    CHECK_EQ(shtest4[2], 0) << "error with shtest4[2]";
    CHECK_EQ(shtest4[3], 0) << "error with shtest4[3]";
    shtest4 = Shape::minusone(4);
    LOG(INFO) << "|--shape of tensor: " << shtest4[0] << ", " << shtest4[1] << ", " << shtest4[2] << ", " <<shtest4[3];
    CHECK_EQ(shtest4[0], -1) << "error with shtest4[0]";
    CHECK_EQ(shtest4[1], -1) << "error with shtest4[1]";
    CHECK_EQ(shtest4[2], -1) << "error with shtest4[2]";
    CHECK_EQ(shtest4[3], -1) << "error with shtest4[3]";

    //! test Shape operate func()
    Shape shtest5(1, 4);
    Shape shtest6(4, 5);
    //it should be each value >/</= the other value
    LOG(INFO) << "|--shtest5 < shtest6: " << (shtest5 < shtest6);
    CHECK_EQ((shtest5 < shtest6), 1) << "error with (shtest5 < shtest6)";
    LOG(INFO) << "|--shtest5 == shtest6: " << (shtest5 == shtest6);
	CHECK_EQ((shtest5 == shtest6), 0) << "error with (shtest5 == shtest6)";
    LOG(INFO) << "|--shtest5 > shtest6: " << (shtest5 > shtest6);
	CHECK_EQ((shtest5 > shtest6), 0) << "error with (shtest5 > shtest6)";

    Shape shtesttemp;
    LOG(INFO) << "|--shape of tensor: " << shtest6[0] << ", " << shtest6[1];
    shtesttemp = shtest6 - shtest5;
    LOG(INFO) << "|--shape of tensor: " << (shtest6-shtest5)[0] << ", " << (shtest6-shtest5)[1];
	CHECK_EQ((shtest6-shtest5)[0], 3) << "error with (shtest6-shtest5)[0]";
	CHECK_EQ((shtest6-shtest5)[1], 1) << "error with (shtest6-shtest5)[1]";

    shtesttemp = shtest5 + shtest6;
    LOG(INFO) << "|--shape of tensor: " << (shtest5+shtest6)[0] << ", " << (shtest5+shtest6)[1];
	CHECK_EQ((shtest5+shtest6)[0], 5) << "error with (shtest5+shtest6)[0]";
	CHECK_EQ((shtest5+shtest6)[1], 9) << "error with (shtest5+shtest6)[1]";

    //! test tensor start_index() function
    LOG(INFO) << "|--test tensor start_index() function" << "," << thost0.start_index();

    //! test tensor get_stride() function
    LOG(INFO) << "|--test tensor get_stride() function";
    Shape shtest7= thost0.get_stride();
    LOG(INFO) << "|--stride of tensor: " << shtest7[0] << ", " << shtest7[1] << "," << shtest7[2] << "," <<shtest7[3];
    LOG(INFO) << "-----------------------sh0.dims(): " << shtest7.dims();
	CHECK_EQ(shtest7.dims(), 4) << "error with shtest7.dims()";
    LOG(INFO) << "-----------------------sh0.count():" << shtest7.count();
	CHECK_EQ(shtest7.count(), 300000) << "error with shtest7.count()";
    LOG(INFO) << "-----------------------sh0.size():" << shtest7.size();
	CHECK_EQ(shtest7.size(), 4) << "error with shtest7.size()";

    //*************NEW******************


    CHECK_EQ(thost0.size(), 600) << "error with tensor size";
    CHECK_EQ(tdev0.size(), 600) << "error with tensor size";

    //! test tensor re_alloc function on tensor with data
    LOG(INFO) << "|--test tensor re_alloc function on tensor with data";
    Shape sh1(1, 3, 10, 10);
    thost0.re_alloc(sh1);
    tdev0.re_alloc(sh1);
    LOG(INFO) << "|--tensor size of host: " << thost0.size();
    LOG(INFO) << "|--tensor size of device: " << tdev0.size();
    CHECK_EQ(thost0.size(), 300) << "error with tensor size";
    CHECK_EQ(tdev0.size(), 300) << "error with tensor size";

    //! test tensor shape() function
    LOG(INFO) << "|--test tensor shape() function";
    Shape sho = thost0.shape();
    LOG(INFO) << "|--shape of tensor: " << sho[0] << ", " << sho[1] << "," << sho[2] << "," <<sho[3];
    LOG(INFO) << "|--test get tensor n, c, h, w function, num = " \
        << thost0.num() << ", channel = " << thost0.channel() << ", height = " \
        << thost0.height() << ", width = " << thost0.width();

    //! test tensor mutable_data() function
    LOG(INFO) << "|--test tensor mutable_data() function, write tensor data buffer with 1.f";
    fill_tensor_host_const(thost0, 1.f);
    LOG(INFO) << "|--test tensor data() function, show the const data, 1.f";
    print_tensor_host(thost0);

    //! test tensor constructor with shape
    LOG(INFO) << "test tensor constructor with shape";
    TensorHf4 thost1(sh1);
    TensorDf4 tdev1(sh1);

    //! test tensor copy_from() function
    LOG(INFO) << "test copy_from() function, input tensor could be any target";
    thost1.copy_from(thost0);
    tdev1.copy_from(thost0);
    print_tensor_device(tdev1);
    cudaDeviceSynchronize();
    thost1.copy_from(tdev1);
    tdev1.copy_from(tdev0);
    print_tensor_host(thost1);

    //! test tensor constructor with shape and real_shape
    LOG(INFO) << "test tensor constructor with shape and real_shape";
    TensorHf4 thost2(sh0, sh1, Shape(0, 0, 0, 0));
    TensorDf4 tdev2(sh0, sh1, Shape(0, 0, 0, 0));

    //! test tensor constructor with data, if target is different, create buffer, and copy the data
    LOG(INFO) << "test tensor constructor with data, if target is different, create buffer, and copy the data";
    dtype* host_data_ptr;
    dtype* dev_data_ptr;
    void *tmp_pt_host;
    void *tmp_pt_dev;
    
    NVH_API::mem_alloc(&tmp_pt_host, sizeof(dtype) * sh1.count());
    host_data_ptr = static_cast<dtype *>(tmp_pt_host);
    for (int i = 0; i < sh1.count(); ++i) {
        host_data_ptr[i] = i;
    }
    NV_API::mem_alloc(&tmp_pt_dev, sizeof(dtype) * sh1.count());
    dev_data_ptr = static_cast<dtype *>(tmp_pt_dev);
    cudaMemcpy(dev_data_ptr, host_data_ptr, sizeof(dtype) * sh1.count(), cudaMemcpyHostToDevice);
    LOG(INFO) << "|--construct host tensor from host data ptr";
    TensorHf4 thost3(host_data_ptr, X86(), NVH_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorDf4 tdev3(host_data_ptr, X86(), NVH_API::get_device_id(), sh1);
    print_tensor_host(thost3);
    print_tensor_device(tdev3);
    cudaDeviceSynchronize();
    
    //! test NVH_API::mem_free()
    LOG(INFO) << "|-- test NVH_API::mem_free()";
    NVH_API::mem_free(host_data_ptr);
     
    LOG(INFO) << "|--construct host tensor from device data ptr";
    TensorHf4 thost4(dev_data_ptr, NV(), NV_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from device data ptr";
    TensorDf4 tdev4(dev_data_ptr, NV(), NV_API::get_device_id(), sh1);
    print_tensor_host(thost4);
    print_tensor_device(tdev4);
    NV_API::stream_t dev_stream0;
    NV_API::create_stream_with_flag(dev_stream0, 1);
    cudaDeviceSynchronize();

    //! test NV_API::mem_set
    LOG(INFO) << "|--test NV_API::mem_set:";
    dtype* host_data_ptrw;
    dtype* dev_data_ptrw;
    void *tmp_pt_hostw;
    void *tmp_pt_devw;
     
    NVH_API::mem_alloc(&tmp_pt_devw, sizeof(dtype) * sh1.count());
    host_data_ptrw = static_cast<dtype *>(tmp_pt_host);
    for (int i = 0; i < sh1.count(); ++i) {
        host_data_ptrw[i] = i;
    }
    dev_data_ptrw = static_cast<dtype *>(tmp_pt_devw);
    cudaMemcpy(host_data_ptrw, dev_data_ptrw, sizeof(dtype) * sh1.count(), cudaMemcpyDeviceToHost);
    LOG(INFO) << "|--construct host tensor from host data ptr";
    TensorHf4 thost3w(host_data_ptrw, X86(), NVH_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorDf4 tdev3w(host_data_ptrw, X86(), NVH_API::get_device_id(), sh1);
    print_tensor_host(thost3w);
    print_tensor_device(tdev3w);
    cudaDeviceSynchronize();
    
    //! test tensor copy constructor
    LOG(INFO) << "test tensor copy constructor";
    LOG(INFO) << "|--normal copy constructor";
    TensorHf4 thost5(thost4);
    TensorDf4 tdev5(tdev4);

    LOG(INFO) << "|--push back to vector";
    std::vector<TensorHf4> vthost;
    std::vector<TensorDf4> vtdev;
    vthost.push_back(thost0);
    vthost.push_back(thost1);
    vthost.push_back(thost2);
    vthost.push_back(thost3);
    vthost.push_back(thost4);
    vthost.push_back(thost5);
    vtdev.push_back(tdev0);
    vtdev.push_back(tdev1);
    vtdev.push_back(tdev2);
    vtdev.push_back(tdev3);
    vtdev.push_back(tdev4);
    vtdev.push_back(tdev5);
    print_tensor_host(vthost[5]);
    print_tensor_device(vtdev[5]);
    cudaDeviceSynchronize();

    //! test share_from function, if targets are the same, buffer is shared, otherwise, buffer is copied
    LOG(INFO) << "test share_from function";
    TensorHf4 thost6, thost7;
    TensorDf4 tdev6, tdev7;
    thost6.set_shape(thost4.shape());
    thost7.set_shape(thost4.shape());
    tdev6.set_shape(thost4.shape());
    tdev7.set_shape(thost4.shape());
    Shape sh2(1, 3, 5, 5);
    Shape offset(0, 0, 5, 5);
    LOG(INFO) << "|--shared host";
    thost6.share_from(thost4);
    LOG(INFO) << "|--copied host";
    tdev6.share_from(thost4);
    LOG(INFO) << "|--copied device";
    thost7.share_from(tdev4);
    LOG(INFO) << "|--shared device";
    tdev7.share_from(tdev4);
    
    LOG(INFO) << "|--change data in shared tensor";

    Shape sh_real = thost6.shape();
    Shape sh_act = thost6.valid_shape();
    Shape offset_act = thost6.offset();

    int start_w = offset_act[3];
    int start_h = offset_act[2];
    int start_c = offset_act[1];
    int start_n = offset_act[0];
    int stride_h = sh_real.count(3);
    int stride_c = sh_real.count(2);
    int stride_n = sh_real.count(1);
    //int stride_n = sh_real.count(0);
    int w = thost6.width();
    int h = thost6.height();
    int c = thost6.channel();
    int n = thost6.num();

    dtype* ptr_host = thost6.mutable_data();
    for (int in = 0; in < n; ++in) {
        dtype* ptr_batch = ptr_host + (in + start_n) * stride_n;
        for (int ic = 0; ic < c; ++ic) {
            dtype* ptr_channel = ptr_batch + (ic + start_c) * stride_c;
            for (int ih = 0; ih < h; ++ih) {
                dtype* ptr_row = ptr_channel + (ih + start_h) * stride_h;
                for (int iw = 0; iw < w; ++iw) {
                    ptr_row[start_w + iw] = 1.f;
                }
            }
        }
    }

    LOG(INFO) << "|--show root tensor while data is changed by shared tensor";
    print_tensor_host(thost4);
    
    //! test record tensor event
    LOG(INFO) << "|--test record tensor event";
    LOG(INFO) << "|--test create stream in different function:";
    NV_API::stream_t dev_stream;
    NV_API::stream_t dev_stream1;
    NV_API::stream_t dev_stream2;
    NV_API::event_t event;
    NV_API::create_event(event,true);   

    NV_API::create_stream_with_flag(dev_stream, 1);
    NV_API::create_stream(dev_stream1);
    NV_API::create_stream_with_priority(dev_stream2, 1, 1);
    
    NVH_API::stream_t host_stream;
    NVH_API::create_stream_with_flag(host_stream, 1);
    LOG(INFO) << "|--test record event on host tensor";
    fill_tensor_host_const(thost4, 888.f);
    thost4.record_event(host_stream);
    thost4.sync();
    print_tensor_host(thost4);
    
    LOG(INFO) << "|--test record event on device tensor";
    fill_tensor_device_const(tdev4, 666.f, dev_stream);
    
    tdev4.record_event(dev_stream);
    tdev4.sync();
    
    // ! test sync_stream function
    LOG(INFO) << "|--test sync_stream function:";
    NV_API::sync_stream(event,dev_stream);
    print_tensor_device(tdev4, dev_stream1);
    tdev4.record_event(dev_stream1);
    tdev4.sync();
    NV_API::destroy_stream(dev_stream1);
    // ! test sync_memcpy function
    LOG(INFO) << "|--test------------------";
    dtype* host_data_ptr2;
    dtype* dev_data_ptr2;
    void *tmp_pt_host2;
    void *tmp_pt_dev2;
    NVH_API::mem_alloc(&tmp_pt_host2, sizeof(dtype) * sh1.count());
    host_data_ptr2 = static_cast<dtype *>(tmp_pt_host2);

    NV_API::mem_alloc(&tmp_pt_dev2, sizeof(dtype) * sh1.count());
    dev_data_ptr2 = static_cast<dtype *>(tmp_pt_dev2);
    
    LOG(INFO) << "|--test cuda, async, P2P" ;
    NV_API::async_memcpy_p2p(host_data_ptr2, 1, dev_data_ptr2, 0, sizeof(dtype) * sh1.count(), dev_stream) ;
    //cudaMemcpy(host_data_ptrw, dev_data_ptrw, sizeof(dtype) * sh1.count(), cudaMemcpyDeviceToHost);
    TensorHf4 thost8(host_data_ptr2, X86(), NVH_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorDf4 tdev8(host_data_ptr2, X86(), NVH_API::get_device_id(), sh1);
    print_tensor_host(thost8);
    LOG(INFO) << "|--test cuda, sync, P2P" ;
    NV_API::sync_memcpy_p2p(host_data_ptr2, 1, dev_data_ptr2, 0, sizeof(dtype) * sh1.count()) ;
    LOG(INFO) << "cuda, async, memcpy" ;
 
    //print_tensor_device(tdev8);
    cudaDeviceSynchronize();
}


int main(int argc, const char** argv){
    // initial logger
    Env<NVHX86>::env_init();
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

