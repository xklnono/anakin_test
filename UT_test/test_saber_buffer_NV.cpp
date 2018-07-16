#include "test_saber_buffer_NV.h"
#include "saber_types.h"
#include "common.h"

using namespace anakin::saber;


template <DataType datatype>
void test_buffer(){

    typedef TargetWrapper<X86> X86_API;
    typedef TargetWrapper<NV> NV_API;
    typedef typename DataTrait<datatype>::dtype Dtype;
    typedef Buffer<X86, datatype> BufferH;
    typedef Buffer<NV, datatype> BufferD;

    int n0 = 1024;
    int n1 = 2048;

    void* tmp_x86;
    Dtype* x86_ptr;
    X86_API::mem_alloc(&tmp_x86, sizeof(Dtype) * n0);
    x86_ptr = static_cast<Dtype*>(tmp_x86);
    for(int i = 0; i < n0; i++){
        x86_ptr[i] = static_cast<Dtype>(i);
    }

    void* tmp_nv;
    Dtype* nv_ptr;
    NV_API::mem_alloc(&tmp_nv, sizeof(Dtype) * n0);
    nv_ptr = static_cast<Dtype*>(tmp_nv);

    LOG(INFO) << "|--Buffer: test default(empty) constructor";
    BufferH x86_buf0;
    BufferD nv_buf0;

    LOG(INFO) << "|--Buffer: test constructor with data size";
    BufferH x86_buf1(n0);
    BufferD nv_buf1(n0);

    LOG(INFO) << "|--Buffer: test constructor with data pointer, size and device id";
    BufferH x86_buf2(x86_ptr, n0, X86_API::get_device_id());
    BufferD nv_buf2(nv_ptr, n0, NV_API::get_device_id());

    LOG(INFO) << "|--Buffer: test copy constructor";
    BufferH x86_buf3(x86_buf2);
    LOG(INFO) << "|--NV Buffer copy constructor";
    LOG(INFO) << "|--nv target id: " << NV_API::get_device_id();
    LOG(INFO) << "|--nv buffer target id: " << nv_buf2.get_id();
    BufferD nv_buf3(nv_buf2);
    CHECK_EQ(x86_buf3.get_count(), x86_buf2.get_count()) << "shared buffer should have same data count";
    CHECK_EQ(nv_buf3.get_count(), nv_buf2.get_count()) << "shared buffer should have same data count";
    //CHECK_EQ(x86_buf3.get_data()[n0 / 2], x86_buf2.get_data()[n0 / 2]) << "shared buffer should have same data value";
    //CHECK_EQ(nv_buf3.get_data()[n0 / 2], nv_buf2.get_data()[n0 / 2]) << "shared buffer should have same data value";

    LOG(INFO) << "|--Buffer: test operator =";
    x86_buf0 = x86_buf2;
    nv_buf0 = nv_buf2;
    CHECK_EQ(x86_buf0.get_count(), x86_buf2.get_count()) << "shared buffer should have same data count";
    CHECK_EQ(nv_buf0.get_count(), nv_buf2.get_count()) << "shared buffer should have same data count";
    //CHECK_EQ(x86_buf0.get_data()[n0 / 2], x86_buf2.get_data()[n0 / 2]) << "shared buffer should have same data value";
    //CHECK_EQ(nv_buf0.get_data()[n0 / 2], nv_buf2.get_data()[n0 / 2]) << "shared buffer should have same data value";

    //! test buffer shared_from() when _id == get_device_id()
    BufferH x86_buf_test0;
    int result_1 = x86_buf_test0.shared_from(x86_buf2);
    LOG(INFO) << "|--Buffer: test buffer result ?= 1 : " << result_1;
    //! TODO
    //! test buffer shared_from() when _id != get_device_id()
    //BufferD nv_buf_test0(n0);
    //int result_0 = nv_buf_test0.shared_from(nv_buf1);
    //LOG(INFO) << "-----------------Buffer: test buffer result ?= 0 : " << result_0;

    //! test buffer mem_set() when _own_data == true
    BufferH x86_buf_test1;
    SaberStatus status = x86_buf_test1.mem_set(0, 0);
    std::string status_info = saber_get_error_string(status);
    LOG(INFO) << "|--Buffer: test buffer mem_set: " << status_info;
    //_own_data == false
    BufferH x86_buf_test2(x86_ptr, n0, X86_API::get_device_id());
    status = x86_buf_test2.mem_set(10, 2048);
    status_info = saber_get_error_string(status);
    LOG(INFO) << "|--Buffer: test buffer mem_set: " << status_info;

    //! test saber_get_error_string() in saber/core/common.h 
    SaberStatus success_status = 0;
    status_info = saber_get_error_string(success_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus not_initialized_status = 1;
    status_info = saber_get_error_string(not_initialized_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus invalid_value_status = 2;
    status_info = saber_get_error_string(invalid_value_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus memalloc_failed_status = 3;
    status_info = saber_get_error_string(memalloc_failed_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus unknown_status = 4;
    status_info = saber_get_error_string(unknown_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus out_of_authorith_status = 5;
    status_info = saber_get_error_string(out_of_authorith_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus out_of_memory_status = 6;
    status_info = saber_get_error_string(out_of_memory_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus unimpl_error_status = 7;
    status_info = saber_get_error_string(unimpl_error_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;
    SaberStatus not_exist_status = 8;
    status_info = saber_get_error_string(not_exist_status);
    LOG(INFO) << "|--Buffer: test no param status_info: " << status_info;

    LOG(INFO) << "|--Buffer: test re_alloc";
    x86_buf1.re_alloc(n1);
    nv_buf1.re_alloc(n1);
    CHECK_EQ(x86_buf1.get_count(), n1) << "buffer count error";
    CHECK_EQ(x86_buf1.get_capacity(), n1) << "buffer capacity error";
    CHECK_EQ(nv_buf1.get_count(), n1) << "buffer count error";
    CHECK_EQ(nv_buf1.get_capacity(), n1) << "buffer capacity error";
    x86_buf1.re_alloc(n0);
    nv_buf1.re_alloc(n0);
    CHECK_EQ(x86_buf1.get_count(), n0) << "buffer count error";
    CHECK_EQ(x86_buf1.get_capacity(), n1) << "buffer capacity error";
    CHECK_EQ(x86_buf1.get_count(), n0) << "buffer count error";
    CHECK_EQ(x86_buf1.get_capacity(), n1) << "buffer capacity error";

    LOG(INFO) << "|--Buffer: test get_id()";
    LOG(INFO) << "|--X86 device id: " << x86_buf0.get_id() << \
        ", nv device id: " << nv_buf0.get_id();
    CHECK_EQ(X86_API::get_device_id(), x86_buf0.get_id()) << "x86 device id error";
    CHECK_EQ(NV_API::get_device_id(), nv_buf0.get_id()) << "nv device id error";

    LOG(INFO) << "|--Buffer: test deep_cpy()";
    x86_buf1.sync_copy_from(x86_buf2);
    LOG(INFO) << "|--deep copy between two host buffer: ";
    for(int i = 0; i < 10;i++) {
        std::cout << x86_buf1.get_data()[i] << std::endl;
    }
    CHECK_EQ(x86_buf1.get_data()[n0 / 2], x86_buf2.get_data()[n0 / 2]) << "deep copy between host is incorrect";
    LOG(INFO) << "|--deep copy from host buffer to device buffer";
    nv_buf1.sync_copy_from(x86_buf2);
    x86_buf1.sync_copy_from(nv_buf1);
    LOG(INFO) << "|--deep copy from device buffer to host buffer: ";
    for(int i = 0; i < 10;i++) {
        std::cout << x86_buf1.get_data()[i] << std::endl;
    }
}

TEST(TestSaberBufferNV, test_buffer_memcpy) {
    test_buffer<AK_FLOAT>();
}

int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
