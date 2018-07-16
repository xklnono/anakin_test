#include "saber_types.h"
#include "target_wrapper.h"
#include <iostream>

#ifdef USE_CUDA
using namespace anakin::saber;
int main(){
	typedef TargetWrapper<NV> API;
	float* ptr;
    //! test mem_free
	API::mem_free(ptr);

    //! test mem_set
	API::mem_set(ptr, 44, 10);

    //! TODO
    //! this functions have no define
	//API::create_stream(stream);
    //API::create_stream_with_priority(stream, flag, priority);
    //API::destroy_stream(stream)
    //API::sync_stream(event, stream)
    //API::async_memcpy_p2p(Dtype* dst, int dst_dev, const Dtype* src, \
    //    int src_dev, size_t count, stream_t& stream)
    //API::sync_memcpy_p2p(Dtype* dst, int dst_dev, const Dtype* src, \
    //    int src_dev, size_t count)

}
#endif

