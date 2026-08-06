

#include "cuda_blackboard/cuda_pointcloud2.hpp"
#include "cuda_blackboard/copy_tracker.hpp"

#include <rclcpp/rclcpp.hpp>

#include <cuda_runtime_api.h>

namespace cuda_blackboard
{

namespace
{

CudaPointCloud2::DeviceDataPtr make_owning_cuda_ptr(std::size_t bytes)
{
  std::uint8_t * ptr = nullptr;
  CUDA_BLACKBOARD_CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&ptr), bytes));
  return CudaPointCloud2::DeviceDataPtr(
    ptr, [](std::uint8_t * p) {
      if (p) {
        CUDA_BLACKBOARD_CHECK_CUDA_ERROR(cudaFree(p));
      }
    });
}

CudaPointCloud2::DeviceDataPtr make_non_owning_cuda_ptr(std::uint8_t * ptr)
{
  return CudaPointCloud2::DeviceDataPtr(ptr, [](std::uint8_t *) {});
}

void copy_metadata(sensor_msgs::msg::PointCloud2 & destination, const sensor_msgs::msg::PointCloud2 & source)
{
  destination.header = source.header;
  destination.fields = source.fields;
  destination.height = source.height;
  destination.width = source.width;
  destination.row_step = source.row_step;
  destination.point_step = source.point_step;
  destination.is_dense = source.is_dense;
  destination.is_bigendian = source.is_bigendian;
}

}  // namespace

CudaPointCloud2::CudaPointCloud2()
: data(nullptr, [](std::uint8_t *) {})
{
}

CudaPointCloud2::CudaPointCloud2(CudaPointCloud2 && source)
{
  copy_metadata(*this, source);
  data = std::move(source.data);
  device_memory_lease_ = std::move(source.device_memory_lease_);

  source.data = nullptr;
}

CudaPointCloud2 & CudaPointCloud2::operator=(CudaPointCloud2 && other)
{
  if (this != &other) {
    if (device_memory_lease_) {
      data.release();
      device_memory_lease_.release();
    }
    copy_metadata(*this, other);
    data = std::move(other.data);
    device_memory_lease_ = std::move(other.device_memory_lease_);

    other.data = nullptr;
  }

  return *this;
}

CudaPointCloud2::CudaPointCloud2(const CudaPointCloud2 & pointcloud)
: sensor_msgs::msg::PointCloud2(pointcloud)
{
  RCLCPP_WARN(
    rclcpp::get_logger("CudaPointCloud2"),
    "CudaPointCloud2 copy constructor called. This should be avoided and is most likely a design "
    "error.");

  data = make_owning_cuda_ptr(pointcloud.height * pointcloud.width * pointcloud.point_step * sizeof(uint8_t));
  tracked_cuda_memcpy(
    data.get(), pointcloud.data.get(),
    pointcloud.height * pointcloud.width * pointcloud.point_step * sizeof(uint8_t),
    cudaMemcpyDeviceToDevice, "CudaPointCloud2 copy constructor");
}

CudaPointCloud2::CudaPointCloud2(const sensor_msgs::msg::PointCloud2 & source)
{
  copy_metadata(*this, source);

  data = make_owning_cuda_ptr(source.height * source.width * source.point_step * sizeof(uint8_t));
  tracked_cuda_memcpy(
    data.get(), source.data.data(), source.height * source.width * source.point_step * sizeof(uint8_t),
    cudaMemcpyHostToDevice, "CudaPointCloud2 from ROS constructor");
}

CudaPointCloud2::CudaPointCloud2(
  const sensor_msgs::msg::PointCloud2 & metadata,
  GpuPointCloudMemoryPool::Lease && device_memory_lease)
: device_memory_lease_(std::move(device_memory_lease))
{
  if (!device_memory_lease_) {
    throw std::invalid_argument("CudaPointCloud2 requires a valid GPU memory lease");
  }

  copy_metadata(*this, metadata);
  const std::size_t bytes = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) *
                            static_cast<std::size_t>(point_step);
  if (bytes > device_memory_lease_.slot().device_capacity_bytes) {
    throw std::invalid_argument("CudaPointCloud2 metadata exceeds leased GPU memory capacity");
  }

  data = make_non_owning_cuda_ptr(device_memory_lease_.slot().device_data);
}

CudaPointCloud2::~CudaPointCloud2()
{
  if (device_memory_lease_) {
    data.release();
  }
}

std::size_t CudaPointCloud2::device_capacity_bytes() const
{
  if (device_memory_lease_) {
    return device_memory_lease_.slot().device_capacity_bytes;
  }
  return static_cast<std::size_t>(height) * static_cast<std::size_t>(width) *
         static_cast<std::size_t>(point_step);
}

bool CudaPointCloud2::is_pool_backed() const
{
  return static_cast<bool>(device_memory_lease_);
}

}  // namespace cuda_blackboard
