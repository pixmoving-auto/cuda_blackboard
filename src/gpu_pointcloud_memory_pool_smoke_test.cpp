#include "cuda_blackboard/gpu_pointcloud_memory_pool.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

int main()
{
  cuda_blackboard::GpuPointCloudMemoryPoolConfig config;
  config.slot_count = 2;
  config.device_slot_bytes = 1024 * 1024;
  config.host_slot_bytes = 1024 * 1024;
  config.allocate_host_slots = true;

  cuda_blackboard::GpuPointCloudMemoryPool pool(config);
  if (pool.slot_count() != 2 || pool.available_count() != 2) {
    throw std::runtime_error("unexpected initial pool state");
  }

  auto first = pool.try_acquire();
  auto second = pool.try_acquire();
  auto third = pool.try_acquire();
  if (!first || !second || third || pool.available_count() != 0) {
    throw std::runtime_error("unexpected acquire behavior");
  }

  cudaMemset(first->slot().device_data, 0x5a, first->slot().device_capacity_bytes);
  cudaDeviceSynchronize();

  first->release();
  if (pool.available_count() != 1) {
    throw std::runtime_error("slot was not released");
  }

  auto reacquired = pool.try_acquire();
  if (!reacquired || pool.available_count() != 0) {
    throw std::runtime_error("released slot was not reusable");
  }

  std::cout << "GpuPointCloudMemoryPool smoke test passed: slots=" << pool.slot_count()
            << " device_slot_bytes=" << pool.device_slot_bytes()
            << " host_slot_bytes=" << pool.host_slot_bytes() << std::endl;
  return 0;
}
