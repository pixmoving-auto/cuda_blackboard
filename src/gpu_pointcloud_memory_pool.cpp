#include "cuda_blackboard/gpu_pointcloud_memory_pool.hpp"

#include "cuda_blackboard/cuda_unique_ptr.hpp"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace cuda_blackboard
{

namespace
{

void validate_config(const GpuPointCloudMemoryPoolConfig & config)
{
  if (config.slot_count == 0) {
    throw std::invalid_argument("GpuPointCloudMemoryPool slot_count must be greater than zero");
  }
  if (config.device_slot_bytes == 0) {
    throw std::invalid_argument("GpuPointCloudMemoryPool device_slot_bytes must be greater than zero");
  }
  if (config.allocate_host_slots && config.host_slot_bytes == 0) {
    throw std::invalid_argument(
      "GpuPointCloudMemoryPool host_slot_bytes must be greater than zero when host slots are enabled");
  }
}

}  // namespace

GpuPointCloudMemoryPool::Lease::Lease(
  GpuPointCloudMemoryPool * pool, GpuPointCloudMemorySlot slot)
: pool_(pool), slot_(slot)
{
}

GpuPointCloudMemoryPool::Lease::Lease(Lease && other) noexcept
: pool_(std::exchange(other.pool_, nullptr)), slot_(other.slot_)
{
}

GpuPointCloudMemoryPool::Lease & GpuPointCloudMemoryPool::Lease::operator=(Lease && other) noexcept
{
  if (this != &other) {
    release();
    pool_ = std::exchange(other.pool_, nullptr);
    slot_ = other.slot_;
  }
  return *this;
}

GpuPointCloudMemoryPool::Lease::~Lease()
{
  release();
}

GpuPointCloudMemoryPool::Lease::operator bool() const
{
  return pool_ != nullptr;
}

const GpuPointCloudMemorySlot & GpuPointCloudMemoryPool::Lease::slot() const
{
  if (!pool_) {
    throw std::runtime_error("GpuPointCloudMemoryPool lease is not valid");
  }
  return slot_;
}

GpuPointCloudMemorySlot & GpuPointCloudMemoryPool::Lease::slot()
{
  if (!pool_) {
    throw std::runtime_error("GpuPointCloudMemoryPool lease is not valid");
  }
  return slot_;
}

void GpuPointCloudMemoryPool::Lease::release()
{
  if (pool_) {
    pool_->release(slot_.index);
    pool_ = nullptr;
  }
}

GpuPointCloudMemoryPool::GpuPointCloudMemoryPool(const GpuPointCloudMemoryPoolConfig & config)
: config_(config)
{
  validate_config(config_);
  slots_.resize(config_.slot_count);
  in_use_.resize(config_.slot_count, false);

  try {
    for (std::size_t i = 0; i < config_.slot_count; ++i) {
      slots_[i].index = i;
      slots_[i].device_capacity_bytes = config_.device_slot_bytes;
      CUDA_BLACKBOARD_CHECK_CUDA_ERROR(
        cudaMalloc(reinterpret_cast<void **>(&slots_[i].device_data), config_.device_slot_bytes));

      if (config_.allocate_host_slots) {
        slots_[i].host_capacity_bytes = config_.host_slot_bytes;
        CUDA_BLACKBOARD_CHECK_CUDA_ERROR(
          cudaMallocHost(reinterpret_cast<void **>(&slots_[i].host_data), config_.host_slot_bytes));
      }
    }
  } catch (...) {
    for (auto & slot : slots_) {
      if (slot.host_data) {
        cudaFreeHost(slot.host_data);
      }
      if (slot.device_data) {
        cudaFree(slot.device_data);
      }
    }
    throw;
  }
}

GpuPointCloudMemoryPool::~GpuPointCloudMemoryPool()
{
  for (auto & slot : slots_) {
    if (slot.host_data) {
      cudaFreeHost(slot.host_data);
      slot.host_data = nullptr;
    }
    if (slot.device_data) {
      cudaFree(slot.device_data);
      slot.device_data = nullptr;
    }
  }
}

std::optional<GpuPointCloudMemoryPool::Lease> GpuPointCloudMemoryPool::try_acquire()
{
  std::scoped_lock lock(mutex_);
  for (std::size_t i = 0; i < in_use_.size(); ++i) {
    if (!in_use_[i]) {
      in_use_[i] = true;
      return Lease(this, slots_[i]);
    }
  }
  return std::nullopt;
}

std::optional<GpuPointCloudMemoryPool::Lease> GpuPointCloudMemoryPool::try_acquire(
  std::size_t min_device_bytes)
{
  if (min_device_bytes > config_.device_slot_bytes) {
    return std::nullopt;
  }
  return try_acquire();
}

std::size_t GpuPointCloudMemoryPool::slot_count() const
{
  return config_.slot_count;
}

std::size_t GpuPointCloudMemoryPool::available_count() const
{
  std::scoped_lock lock(mutex_);
  std::size_t count = 0;
  for (const bool used : in_use_) {
    if (!used) {
      ++count;
    }
  }
  return count;
}

std::size_t GpuPointCloudMemoryPool::device_slot_bytes() const
{
  return config_.device_slot_bytes;
}

std::size_t GpuPointCloudMemoryPool::host_slot_bytes() const
{
  return config_.allocate_host_slots ? config_.host_slot_bytes : 0;
}

void GpuPointCloudMemoryPool::release(std::size_t index)
{
  std::scoped_lock lock(mutex_);
  if (index >= in_use_.size() || !in_use_[index]) {
    std::stringstream message;
    message << "GpuPointCloudMemoryPool invalid release for slot " << index;
    throw std::runtime_error(message.str());
  }
  in_use_[index] = false;
}

}  // namespace cuda_blackboard
