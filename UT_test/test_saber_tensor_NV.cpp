#include "test_saber_tensor_NV.h"
#include "tensor_op.h"
#include <vector>
#include "core/target_traits.h"
using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
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
    X86_API::mem_alloc(&tmp_pt_host, sizeof(dtype) * sh1.count());
    host_data_ptr = static_cast<dtype *>(tmp_pt_host);
    for (int i = 0; i < sh1.count(); ++i) {
        host_data_ptr[i] = i;
    }
    NV_API::mem_alloc(&tmp_pt_dev, sizeof(dtype) * sh1.count());
    dev_data_ptr = static_cast<dtype *>(tmp_pt_dev);
    cudaMemcpy(dev_data_ptr, host_data_ptr, sizeof(dtype) * sh1.count(), cudaMemcpyHostToDevice);
    LOG(INFO) << "|--construct host tensor from host data ptr";
    TensorHf4 thost3(host_data_ptr, X86(), X86_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorDf4 tdev3(host_data_ptr, X86(), X86_API::get_device_id(), sh1);
    print_tensor_host(thost3);
    print_tensor_device(tdev3);
    cudaDeviceSynchronize();
    
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
    LOG(INFO) << "-------------------test mem_set:";
    dtype* host_data_ptrw;
    dtype* dev_data_ptrw;
    void *tmp_pt_hostw;
    void *tmp_pt_devw;
    X86_API::mem_alloc(&tmp_pt_hostw, sizeof(dtype) * sh1.count());
    host_data_ptrw = static_cast<dtype *>(tmp_pt_hostw);
    
    NV_API::mem_alloc(&tmp_pt_devw, sizeof(dtype) * sh1.count());
    dev_data_ptrw = static_cast<dtype *>(tmp_pt_devw);
    cudaMemcpy(host_data_ptrw, dev_data_ptrw, sizeof(dtype) * sh1.count(), cudaMemcpyDeviceToHost);
    LOG(INFO) << "|--construct host tensor from host data ptr";
    TensorHf4 thost3w(host_data_ptrw, X86(), X86_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorDf4 tdev3w(host_data_ptrw, X86(), X86_API::get_device_id(), sh1);
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
    
    X86_API::stream_t host_stream;
    X86_API::create_stream_with_flag(host_stream, 1);
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
    X86_API::mem_alloc(&tmp_pt_host2, sizeof(dtype) * sh1.count());
    host_data_ptr2 = static_cast<dtype *>(tmp_pt_host2);

    NV_API::mem_alloc(&tmp_pt_dev2, sizeof(dtype) * sh1.count());
    dev_data_ptr2 = static_cast<dtype *>(tmp_pt_dev2);
    
    LOG(INFO) << "|--test cuda, async, P2P" ;
    NV_API::async_memcpy_p2p(host_data_ptr2, 1, dev_data_ptr2, 0, sizeof(dtype) * sh1.count(), dev_stream) ;
    //cudaMemcpy(host_data_ptrw, dev_data_ptrw, sizeof(dtype) * sh1.count(), cudaMemcpyDeviceToHost);
    TensorHf4 thost8(host_data_ptr2, X86(), X86_API::get_device_id(), sh1);
    LOG(INFO) << "|--constructor device tensor from host data ptr";
    TensorDf4 tdev8(host_data_ptr2, X86(), X86_API::get_device_id(), sh1);
    print_tensor_host(thost8);
    LOG(INFO) << "|--test cuda, sync, P2P" ;
    NV_API::sync_memcpy_p2p(host_data_ptr2, 1, dev_data_ptr2, 0, sizeof(dtype) * sh1.count()) ;
    LOG(INFO) << "cuda, async, memcpy" ;
 
    //print_tensor_device(tdev8);
    cudaDeviceSynchronize();
}

TEST(TestSaberTensorNV, test_tensor_deepcopy) {
    //! tensor constructor with alloc data, if target is different, create buffer, and copy the data
    LOG(INFO) << "test tensor deep copy";
    Shape sh0(2, 4, 8, 8);
    Shape va_sh0(2, 4, 4, 4);
    Shape off_sh0(0, 0, 2, 2);
    Shape sh1(2, 4, 10, 4);
    Shape va_sh1(va_sh0);
    Shape off_sh1(0, 0, 4, 0);
    Shape sh2(4, 64);
    Shape va_sh2(2, 64);
    Shape off_sh2(1, 0);

    X86_API::stream_t x86_stream;
    NV_API::stream_t nv_stream;
    X86_API::create_stream(x86_stream);
    NV_API::create_stream(nv_stream);

    //! create source tensor, th0, td0, th01, td01, th1, td1;
    TensorHf4 th0(sh0);
    for (int i = 0; i < sh0.count(); ++i) {
        th0.mutable_data()[i] = i;
    }
    TensorHf4 th1(va_sh0);
    for (int i = 0; i < va_sh0.count(); ++i) {
        th1.mutable_data()[i] = i;
    }
    TensorHf4 th01;
    th01.share_sub_buffer(th0, va_sh0, off_sh0);

    TensorDf4 td0, td1, td01;
    td0.set_shape(th0.shape());
    td1.set_shape(th1.shape());
    td0.share_from(th0);
    td1.share_from(th1);
    TensorDf4 dev_tmp0;
    dev_tmp0.set_shape(th0.shape());
    dev_tmp0.share_from(th0);
    td01.share_sub_buffer(dev_tmp0, va_sh0, off_sh0);

    print_tensor_host(th0);
    print_tensor_host(th1);
    print_tensor_device(td0);
    print_tensor_device(td1);

    //! create th2, th3, th21, td2, td3, td21 as dst tensor
    TensorHf2 th2(sh2);
    fill_tensor_host_const(th2, 0.f);
    TensorHf2 th21;
    th21.share_sub_buffer(th2, va_sh2, off_sh2);
    TensorHf2 th3(va_sh2);

    TensorDf2 td2(sh2);
    fill_tensor_device_const(td2, 0.f);
    cudaDeviceSynchronize();
    TensorDf2 td21;
    td21.share_sub_buffer(td2, va_sh2, off_sh2);
    TensorDf2 td3(va_sh2);

    double max_diff;
    double  max_ratio;
    //! test tensor deep copy, entire buffer copy
    LOG(INFO) << "test tensor deep copy, entire buffer copy, H2H";
    th3.copy_from(th1);
    print_tensor_host(th3);
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, H2H";
    fill_tensor_host_const(th3, 0.f);
    th3.async_copy_from(th1, x86_stream);
    th3.record_event(x86_stream);
    th3.sync();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, H2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, D2H";
    th3.copy_from(td1);
    print_tensor_host(th3);
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2H";
    fill_tensor_host_const(th3, 0.f);
    th3.async_copy_from(td1, nv_stream);
    th3.record_event(x86_stream);
    th3.sync();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, H2D";
    td3.copy_from(th1);
    print_tensor_device(td3);
    cudaDeviceSynchronize();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2H";
    fill_tensor_device_const(td3, 0.f);
    cudaDeviceSynchronize();
    td3.async_copy_from(th1, nv_stream);
    td3.record_event(nv_stream);
    td3.sync();
    tensor_cmp_host(th1.data(), th3.data(), th3.size(), max_ratio, max_diff);
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2H";

    LOG(INFO) << "test tensor deep copy, entire buffer copy, D2D";
    td3.copy_from(td1);
    print_tensor_device(td3);
    cudaDeviceSynchronize();
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, sync, D2D";
    fill_tensor_device_const(td3, 0.f);
    cudaDeviceSynchronize();
    td3.async_copy_from(td1, nv_stream);
    td3.record_event(nv_stream);
    td3.sync();
    CHECK_LE(max_ratio, 1e-5f) << "error result of entire buffer copy, async, D2D";


    //! test tensor deep copy, src with roi
    LOG(INFO) << "test tensor deep copy, src with roi, H2H";
    th3.copy_from(th01);
    print_tensor_host(th3);

    LOG(INFO) << "test tensor deep copy, src with roi, D2H";
    th3.copy_from(td01);
    print_tensor_host(th3);

    LOG(INFO) << "test tensor deep copy, src with roi, H2D";
    td3.copy_from(th01);
    print_tensor_device(td3);
    cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, src with roi, D2D";
    td3.copy_from(td01);
    print_tensor_device(td3);
    cudaDeviceSynchronize();


    //! test tensor deep copy, dst with roi
    LOG(INFO) << "test tensor deep copy, dst with roi, H2H";
    print_tensor_host(th21);
    print_tensor_host(th1);
    th21.copy_from(th1);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, dst with roi, D2H";
    th21.copy_from(td1);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, dst with roi, H2D";
    td21.copy_from(th1);
    print_tensor_device(td21);
    cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, dst with roi, D2D";
    td21.copy_from(td1);
    print_tensor_device(td21);
    cudaDeviceSynchronize();


    //! test tensor deep copy, src and dst are with roi
    LOG(INFO) << "test tensor deep copy, src and dst are with roi, H2H";
    th21.copy_from(th01);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, D2H";
    th21.copy_from(td01);
    print_tensor_host(th21);

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, H2D";
    td21.copy_from(th01);
    print_tensor_device(td21);
    cudaDeviceSynchronize();

    LOG(INFO) << "test tensor deep copy, src and dst are with roi, D2D";
    td21.copy_from(td01);
    print_tensor_device(td21);
    cudaDeviceSynchronize();

    TensorDf4 td;
    Shape sh = {1, 3, 10, 10};
    td.re_alloc(sh);
    NV_API::stream_t stream00, stream01;
    NV_API::create_stream(stream00);
    NV_API::create_stream(stream01);
    fill_tensor_device_const(td, 666);
    cudaDeviceSynchronize();
    print_tensor_device(td, stream00);
    td.record_event(stream00);
    //! uncomment the flowing line will print wrong result
    td.sync();
    fill_tensor_device_const(td, 888, stream01);
    cudaDeviceSynchronize();

}

TEST(TestSaberTensorNV, test_tensor_shape) {
    typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4_0;
    typedef Tensor<X86, AK_FLOAT, NHWC> Tensor4_1;
    typedef Tensor<X86, AK_FLOAT, HW> Tensor2;

    int nin = 2;
    int cin = 4;
    int hin = 8;
    int win = 16;

    LOG(INFO) << "test tensor interface";

    Tensor4_0 t1(Shape(nin, cin, hin, win));
    Tensor4_1 t2(Shape(nin, hin, win, cin));
    Tensor2 t3(Shape(hin, win));

    LOG(INFO) << "test tensor with layout of NCHW";
    LOG(INFO) << "num: " << t1.num() << ", num idx: " << t1.num_index() << \
              ", channel: " << t1.channel() << ", channel idx: " << t1.channel_index() << \
              ", height: " << t1.height() << ", height idx: " << t1.height_index() << \
              ", widhth: " << t1.width() << ", width idx: " << t1.width_index();

    CHECK_EQ(t1.num(), nin) << "NCHW get num error";
    CHECK_EQ(t1.channel(), cin) << "NCHW get channel error";
    CHECK_EQ(t1.height(), hin) << "NCHW get height error";
    CHECK_EQ(t1.width(), win) << "NCHW get width error";

    CHECK_EQ(t1.num_index(), 0) << "NCHW get num index error";
    CHECK_EQ(t1.channel_index(), 1) << "NCHW get channel index error";
    CHECK_EQ(t1.height_index(), 2) << "NCHW get height index error";
    CHECK_EQ(t1.width_index(), 3) << "NCHW get width index error";

    LOG(INFO) << "test tensor with layout of NHWC";
    LOG(INFO) << "num: " << t2.num() << ", num idx: " << t2.num_index() << \
              ", channel: " << t2.channel() << ", channel idx: " << t2.channel_index() << \
              ", height: " << t2.height() << ", height idx: " << t2.height_index() << \
              ", widhth: " << t2.width() << ", width idx: " << t2.width_index();

    CHECK_EQ(t2.num(), nin) << "NHWC get num error";
    CHECK_EQ(t2.channel(), cin) << "NHWC get channel error";
    CHECK_EQ(t2.height(), hin) << "NHWC get height error";
    CHECK_EQ(t2.width(), win) << "NHWC get width error";

    CHECK_EQ(t2.num_index(), 0) << "NHWC get num index error";
    CHECK_EQ(t2.channel_index(), 3) << "NHWC get channel index error";
    CHECK_EQ(t2.height_index(), 1) << "NHWC get height index error";
    CHECK_EQ(t2.width_index(), 2) << "NHWC get width index error";

    LOG(INFO) << "test tensor with layout of HW";
    LOG(INFO) << "num: " << t3.num() << ", num idx: " << t3.num_index() << \
              ", channel: " << t3.channel() << ", channel idx: " << t3.channel_index() << \
              ", height: " << t3.height() << ", height idx: " << t3.height_index() << \
              ", widhth: " << t3.width() << ", width idx: " << t3.width_index();

    CHECK_EQ(t3.num(), 1) << "HW get num error";
    CHECK_EQ(t3.channel(), 1) << "HW get channel error";
    CHECK_EQ(t3.height(), hin) << "HW get height error";
    CHECK_EQ(t3.width(), win) << "HW get width error";

    CHECK_EQ(t3.num_index(), -1) << "HW get num index error";
    CHECK_EQ(t3.channel_index(), -1) << "HW get channel index error";
    CHECK_EQ(t3.height_index(), 0) << "HW get height index error";
    CHECK_EQ(t3.width_index(), 1) << "HW get width index error";

}

TEST(TestSaberTensorNV, test_tensor_reshape_realloc) {

    LOG(INFO) << "test tensor reshape and re_alloc funcs";

    Shape sh0(1, 2, 4, 4);
    Shape sh1(1, 2, 8, 8);
    TensorHf4 th0(sh1);
    TensorDf4 td0(sh1);
    fill_tensor_host_const(th0, 1);
    fill_tensor_device_const(td0, 1);
    LOG(INFO) << "ori tensor with size: " << th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    cudaDeviceSynchronize();

    th0.reshape(sh0);
    td0.reshape(sh0);
    LOG(INFO) << "tensor after reshape(from big space to small) with size: " << th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    cudaDeviceSynchronize();
    fill_tensor_host_const(th0, 1);
    fill_tensor_device_const(td0, 1);
    cudaDeviceSynchronize();

    th0.reshape(sh1);
    td0.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small to big, not larger than ori) with size: " << th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    cudaDeviceSynchronize();

    th0.re_alloc(sh0);
    td0.re_alloc(sh0);
    LOG(INFO) << "tensor after re_alloc(from big space to small) with size: " << th0.valid_size();
    print_tensor_host(th0);
    print_tensor_device(td0);
    cudaDeviceSynchronize();

    TensorHf4 th1(sh0);
    TensorDf4 td1(sh0);
    LOG(INFO) << "ori tensor with size: " << th1.valid_size();
    fill_tensor_host_const(th1, 1);
    fill_tensor_device_const(td1, 1);
    cudaDeviceSynchronize();
    print_tensor_host(th1);
    print_tensor_device(td1);
    cudaDeviceSynchronize();

    th1.reshape(sh1);
    td1.reshape(sh1);
    LOG(INFO) << "tensor after reshape(from small space to big) with size: " << th1.valid_size();
    print_tensor_host(th1);
    print_tensor_device(td1);
    cudaDeviceSynchronize();
    fill_tensor_host_const(th1, 1);
    fill_tensor_device_const(td1, 1);
    cudaDeviceSynchronize();

    th1.reshape(sh0);
    td1.reshape(sh0);

    LOG(INFO) << "tensor after re_alloc(from small space to big) with size: " << th1.valid_size();
    th1.re_alloc(sh1);
    td1.re_alloc(sh1);
    print_tensor_host(th1);
    print_tensor_device(td1);
    cudaDeviceSynchronize();

}

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

