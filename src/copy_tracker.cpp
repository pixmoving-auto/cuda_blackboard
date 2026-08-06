#include "cuda_blackboard/copy_tracker.hpp"

#include <rclcpp/rclcpp.hpp>

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

namespace cuda_blackboard
{
namespace
{

struct CopyCounter
{
  std::atomic<std::uint64_t> calls{0};
  std::atomic<std::uint64_t> bytes{0};
};

struct TrackerState
{
  std::array<CopyCounter, 5> counters;
  std::once_flag enabled_once;
  bool enabled{false};

  ~TrackerState()
  {
    if (!enabled) {
      return;
    }
    auto logger = rclcpp::get_logger("cuda_blackboard.copy_tracker");
    RCLCPP_INFO(
      logger,
      "CUDA copy summary: H2D calls=%lu bytes=%lu, D2H calls=%lu bytes=%lu, D2D calls=%lu "
      "bytes=%lu, H2H calls=%lu bytes=%lu, default calls=%lu bytes=%lu",
      counters[cudaMemcpyHostToDevice].calls.load(), counters[cudaMemcpyHostToDevice].bytes.load(),
      counters[cudaMemcpyDeviceToHost].calls.load(), counters[cudaMemcpyDeviceToHost].bytes.load(),
      counters[cudaMemcpyDeviceToDevice].calls.load(),
      counters[cudaMemcpyDeviceToDevice].bytes.load(), counters[cudaMemcpyHostToHost].calls.load(),
      counters[cudaMemcpyHostToHost].bytes.load(), counters[cudaMemcpyDefault].calls.load(),
      counters[cudaMemcpyDefault].bytes.load());
  }
};

TrackerState & state()
{
  static TrackerState instance;
  return instance;
}

bool tracker_enabled()
{
  auto & tracker = state();
  std::call_once(tracker.enabled_once, [&tracker]() {
    const char * value = std::getenv("CUDA_BLACKBOARD_COPY_TRACKING");
    tracker.enabled = value != nullptr && std::string(value) != "0" && std::string(value) != "false";
    if (tracker.enabled) {
      RCLCPP_INFO(
        rclcpp::get_logger("cuda_blackboard.copy_tracker"),
        "CUDA blackboard copy tracking is enabled");
    }
  });
  return tracker.enabled;
}

std::size_t counter_index(cudaMemcpyKind kind)
{
  switch (kind) {
    case cudaMemcpyHostToHost:
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDeviceToDevice:
    case cudaMemcpyDefault:
      return static_cast<std::size_t>(kind);
    default:
      return static_cast<std::size_t>(cudaMemcpyDefault);
  }
}

const char * kind_name(cudaMemcpyKind kind)
{
  switch (kind) {
    case cudaMemcpyHostToHost:
      return "H2H";
    case cudaMemcpyHostToDevice:
      return "H2D";
    case cudaMemcpyDeviceToHost:
      return "D2H";
    case cudaMemcpyDeviceToDevice:
      return "D2D";
    case cudaMemcpyDefault:
      return "default";
    default:
      return "unknown";
  }
}

}  // namespace

cudaError_t tracked_cuda_memcpy(
  void * dst, const void * src, std::size_t count, cudaMemcpyKind kind, const char * label)
{
  if (tracker_enabled()) {
    auto & counter = state().counters[counter_index(kind)];
    counter.calls.fetch_add(1, std::memory_order_relaxed);
    counter.bytes.fetch_add(count, std::memory_order_relaxed);
    RCLCPP_DEBUG(
      rclcpp::get_logger("cuda_blackboard.copy_tracker"), "cudaMemcpy %s bytes=%zu label=%s",
      kind_name(kind), count, label ? label : "unknown");
  }
  return cudaMemcpy(dst, src, count, kind);
}

}  // namespace cuda_blackboard
