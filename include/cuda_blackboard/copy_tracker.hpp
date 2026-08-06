#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuda_blackboard
{

cudaError_t tracked_cuda_memcpy(
  void * dst, const void * src, std::size_t count, cudaMemcpyKind kind, const char * label);

}  // namespace cuda_blackboard
