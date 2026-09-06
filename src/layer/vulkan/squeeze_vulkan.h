// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SQUEEZE_VULKAN_H
#define LAYER_SQUEEZE_VULKAN_H

#include "squeeze.h"

namespace ncnn {

class Squeeze_vulkan : public Squeeze
{
public:
    Squeeze_vulkan();

    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SQUEEZE_VULKAN_H
