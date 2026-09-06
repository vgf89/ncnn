// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "squeeze_vulkan.h"

#include "layer_type.h"

namespace ncnn {

Squeeze_vulkan::Squeeze_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
}

int Squeeze_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    // Zero-copy dim alias, mirroring Squeeze::forward reshapes.
    // Scalarize the (possibly packed) shape, apply the same flag logic,
    // then repack. Bail loudly on layouts we cannot alias exactly.
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    // scalarize packed axis
    if (dims == 1)
        w *= elempack;
    if (dims == 2)
        h *= elempack;
    if (dims == 3 || dims == 4)
        channels *= elempack;

    bool _squeeze_w = false;
    bool _squeeze_h = false;
    bool _squeeze_d = false;
    bool _squeeze_c = false;

    if (axes.empty())
    {
        _squeeze_w = w == 1 && squeeze_w;
        _squeeze_h = h == 1 && squeeze_h;
        _squeeze_d = d == 1 && squeeze_d;
        _squeeze_c = channels == 1 && squeeze_c;
    }
    else
    {
        const int* axes_ptr = axes;
        for (int i = 0; i < axes.w; i++)
        {
            int axis = axes_ptr[i];
            if (axis < 0)
                axis = dims + axis;

            if (dims == 1 && axis == 0)
                _squeeze_w = w == 1;
            if (dims == 2 && axis == 0)
                _squeeze_h = h == 1;
            if (dims == 2 && axis == 1)
                _squeeze_w = w == 1;
            if (dims == 3 && axis == 0)
                _squeeze_c = channels == 1;
            if (dims == 3 && axis == 1)
                _squeeze_h = h == 1;
            if (dims == 3 && axis == 2)
                _squeeze_w = w == 1;
            if (dims == 4 && axis == 0)
                _squeeze_c = channels == 1;
            if (dims == 4 && axis == 1)
                _squeeze_d = d == 1;
            if (dims == 4 && axis == 2)
                _squeeze_h = h == 1;
            if (dims == 4 && axis == 3)
                _squeeze_w = w == 1;
        }
    }

    // scalar output shape, same branches as Squeeze::forward.
    // Every branch sets the full (odims, ow, oh, od, oc); dropped axes are 1.
    int odims = dims;
    int ow = w, oh = h, od = d, oc = channels;
    if (dims == 1 && _squeeze_w)
    {
        odims = 1; ow = 1; oh = 1; od = 1; oc = 1;
    }
    if (dims == 2)
    {
        if (_squeeze_w && _squeeze_h) { odims = 1; ow = 1; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w) { odims = 1; ow = h; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_h) { odims = 1; ow = w; oh = 1; od = 1; oc = 1; }
    }
    if (dims == 3)
    {
        if (_squeeze_w && _squeeze_h && _squeeze_c) { odims = 1; ow = 1; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_h) { odims = 1; ow = channels; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_h && _squeeze_c) { odims = 1; ow = w; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_c) { odims = 1; ow = h; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w) { odims = 2; ow = h; oh = channels; od = 1; oc = 1; }
        else if (_squeeze_h) { odims = 2; ow = w; oh = channels; od = 1; oc = 1; }
        else if (_squeeze_c) { odims = 2; ow = w; oh = h; od = 1; oc = 1; }
    }
    if (dims == 4)
    {
        if (_squeeze_w && _squeeze_h && _squeeze_d && _squeeze_c) { odims = 1; ow = 1; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_h && _squeeze_d) { odims = 1; ow = channels; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_h && _squeeze_d && _squeeze_c) { odims = 1; ow = w; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_d && _squeeze_c) { odims = 1; ow = h; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_h && _squeeze_c) { odims = 1; ow = d; oh = 1; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_h) { odims = 2; ow = d; oh = channels; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_d) { odims = 2; ow = h; oh = channels; od = 1; oc = 1; }
        else if (_squeeze_h && _squeeze_d) { odims = 2; ow = w; oh = channels; od = 1; oc = 1; }
        else if (_squeeze_h && _squeeze_c) { odims = 2; ow = w; oh = d; od = 1; oc = 1; }
        else if (_squeeze_w && _squeeze_c) { odims = 2; ow = h; oh = d; od = 1; oc = 1; }
        else if (_squeeze_d && _squeeze_c) { odims = 2; ow = w; oh = h; od = 1; oc = 1; }
        else if (_squeeze_w) { odims = 3; ow = h; oh = d; od = 1; oc = channels; }
        else if (_squeeze_h) { odims = 3; ow = w; oh = d; od = 1; oc = channels; }
        else if (_squeeze_d) { odims = 3; ow = w; oh = h; od = 1; oc = channels; }
        else if (_squeeze_c) { odims = 3; ow = w; oh = h; od = d; oc = 1; }
    }
    // repack: keep bottom packing, require exact divisibility
    int out_elempack = elempack;
    int outw = ow, outh = oh, outd = od, outc = oc;
    if (odims == 1)
    {
        if (ow % out_elempack != 0) return -1;
        outw = ow / out_elempack;
    }
    if (odims == 2)
    {
        if (oh % out_elempack != 0) return -1;
        outh = oh / out_elempack;
    }
    if (odims == 3 || odims == 4)
    {
        if (oc % out_elempack != 0) return -1;
        outc = oc / out_elempack;
    }
    if ((size_t)ow * oh * od * oc != (size_t)w * h * d * channels)
        return -1;

    top_blob = bottom_blob;
    top_blob.dims = odims;
    top_blob.w = outw;
    top_blob.h = outh;
    top_blob.d = outd;
    top_blob.c = outc;
    {
        size_t outcstep = 0;
        if (odims == 1)
            outcstep = alignSize((size_t)outw * elemsize, 16) / elemsize;
        if (odims == 2)
            outcstep = alignSize((size_t)outw * outh * elemsize, 16) / elemsize;
        if (odims == 3)
            outcstep = alignSize((size_t)outw * outh * elemsize, 16) / elemsize;
        if (odims == 4)
            outcstep = alignSize((size_t)outw * outh * outd * elemsize, 16) / elemsize;
            // pure alias: the new dims must describe the same strided buffer
        if (outcstep * (size_t)outc != bottom_blob.cstep * (size_t)bottom_blob.c)
            return -1;
    }

    (void)cmd;
    (void)opt;
    return 0;
}

} // namespace ncnn
