#include "cuda_blackboard/cuda_pointcloud2.hpp"
#include "cuda_blackboard/gpu_pointcloud_memory_pool.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace
{

sensor_msgs::msg::PointCloud2 make_metadata()
{
  sensor_msgs::msg::PointCloud2 metadata;
  metadata.height = 1;
  metadata.width = 1024;
  metadata.point_step = 16;
  metadata.row_step = metadata.width * metadata.point_step;
  metadata.is_dense = true;
  metadata.fields.resize(4);
  metadata.fields[0].name = "x";
  metadata.fields[0].offset = 0;
  metadata.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  metadata.fields[0].count = 1;
  metadata.fields[1].name = "y";
  metadata.fields[1].offset = 4;
  metadata.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  metadata.fields[1].count = 1;
  metadata.fields[2].name = "z";
  metadata.fields[2].offset = 8;
  metadata.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  metadata.fields[2].count = 1;
  metadata.fields[3].name = "intensity";
  metadata.fields[3].offset = 12;
  metadata.fields[3].datatype = sensor_msgs::msg::PointField::UINT8;
  metadata.fields[3].count = 1;
  return metadata;
}

}  // namespace

int main()
{
  cuda_blackboard::GpuPointCloudMemoryPoolConfig config;
  config.slot_count = 1;
  config.device_slot_bytes = 1024 * 16;

  cuda_blackboard::GpuPointCloudMemoryPool pool(config);
  if (pool.available_count() != 1) {
    throw std::runtime_error("unexpected initial pool availability");
  }

  {
    auto lease = pool.try_acquire(1024 * 16);
    if (!lease) {
      throw std::runtime_error("failed to acquire pool slot");
    }

    cuda_blackboard::CudaPointCloud2 cloud(make_metadata(), std::move(*lease));
    if (!cloud.is_pool_backed() || cloud.device_capacity_bytes() != config.device_slot_bytes) {
      throw std::runtime_error("CudaPointCloud2 is not backed by the pool slot");
    }
    if (pool.available_count() != 0) {
      throw std::runtime_error("pool slot was not retained by CudaPointCloud2");
    }

    cudaMemset(cloud.data.get(), 0x7f, cloud.height * cloud.width * cloud.point_step);
    cudaDeviceSynchronize();
  }

  if (pool.available_count() != 1) {
    throw std::runtime_error("CudaPointCloud2 did not return the pool slot on destruction");
  }

  std::cout << "CudaPointCloud2 pool smoke test passed" << std::endl;
  return 0;
}
