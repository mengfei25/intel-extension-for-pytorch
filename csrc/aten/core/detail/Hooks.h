#pragma once

#include <ATen/Generator.h>
#include <ATen/detail/XPUHooksInterface.h>

namespace xpu {
namespace dpcpp {
namespace detail {

// The real implementation of XPUHooksInterface
struct XPUHooks : public at::XPUHooksInterface {
  XPUHooks(at::XPUHooksArgs) {}
  void initXPU() const override;
  bool hasXPU() const override;
  bool hasOneMKL() const override;
  bool hasOneDNN() const override;
  std::string showConfig() const override;
  int64_t getCurrentDevice() const override;
  int getDeviceCount() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(void* data) const override;
  Allocator* getPinnedMemoryAllocator() const override;
  const Generator& getDefaultXPUGenerator(
      DeviceIndex device_index = -1) const override;
};

} // namespace detail
} // namespace dpcpp
} // namespace xpu
