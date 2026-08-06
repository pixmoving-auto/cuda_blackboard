#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace cuda_blackboard
{

struct GpuPointCloudMemoryPoolConfig
{
  std::size_t slot_count{0};
  std::size_t device_slot_bytes{0};
  std::size_t host_slot_bytes{0};
  bool allocate_host_slots{false};
};

struct GpuPointCloudMemorySlot
{
  std::size_t index{0};
  std::uint8_t * device_data{nullptr};
  std::uint8_t * host_data{nullptr};
  std::size_t device_capacity_bytes{0};
  std::size_t host_capacity_bytes{0};
};

class GpuPointCloudMemoryPool
{
public:
  class Lease
  {
  public:
    Lease() = default;
    Lease(GpuPointCloudMemoryPool * pool, GpuPointCloudMemorySlot slot);
    Lease(const Lease &) = delete;
    Lease & operator=(const Lease &) = delete;
    Lease(Lease && other) noexcept;
    Lease & operator=(Lease && other) noexcept;
    ~Lease();

    explicit operator bool() const;
    const GpuPointCloudMemorySlot & slot() const;
    GpuPointCloudMemorySlot & slot();
    void release();

  private:
    GpuPointCloudMemoryPool * pool_{nullptr};
    GpuPointCloudMemorySlot slot_{};
  };

  explicit GpuPointCloudMemoryPool(const GpuPointCloudMemoryPoolConfig & config);
  GpuPointCloudMemoryPool(const GpuPointCloudMemoryPool &) = delete;
  GpuPointCloudMemoryPool & operator=(const GpuPointCloudMemoryPool &) = delete;
  ~GpuPointCloudMemoryPool();

  std::optional<Lease> try_acquire();
  std::optional<Lease> try_acquire(std::size_t min_device_bytes);
  std::size_t slot_count() const;
  std::size_t available_count() const;
  std::size_t device_slot_bytes() const;
  std::size_t host_slot_bytes() const;

private:
  friend class Lease;

  void release(std::size_t index);

  GpuPointCloudMemoryPoolConfig config_;
  std::vector<GpuPointCloudMemorySlot> slots_;
  std::vector<bool> in_use_;
  mutable std::mutex mutex_;
};

}  // namespace cuda_blackboard
