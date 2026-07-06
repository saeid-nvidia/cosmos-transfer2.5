# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _bilateral_u8_hwc_kernel(
    source,
    destination,
    height: tl.constexpr,
    width: tl.constexpr,
    radius: tl.constexpr,
    sigma_color_sq_x2,
    sigma_space_sq_x2,
    BLOCK_PIXELS: tl.constexpr,
    BLOCK_NEIGHBORS: tl.constexpr,
):
    """Apply an OpenCV-compatible bilateral filter to an HWC RGB image."""
    pixel = tl.program_id(0) * BLOCK_PIXELS + tl.arange(0, BLOCK_PIXELS)
    valid_pixel = pixel < height * width
    y = pixel // width
    x = pixel - y * width

    base = pixel * 3
    center_r = tl.load(source + base, mask=valid_pixel, other=0.0).to(tl.float32)
    center_g = tl.load(source + base + 1, mask=valid_pixel, other=0.0).to(tl.float32)
    center_b = tl.load(source + base + 2, mask=valid_pixel, other=0.0).to(tl.float32)

    sum_r = tl.zeros((BLOCK_PIXELS,), tl.float32)
    sum_g = tl.zeros((BLOCK_PIXELS,), tl.float32)
    sum_b = tl.zeros((BLOCK_PIXELS,), tl.float32)
    sum_weight = tl.zeros((BLOCK_PIXELS,), tl.float32)
    diameter: tl.constexpr = radius * 2 + 1
    window: tl.constexpr = diameter * diameter

    for start in range(0, window, BLOCK_NEIGHBORS):
        neighbor = start + tl.arange(0, BLOCK_NEIGHBORS)
        dy = neighbor // diameter - radius
        dx = neighbor - (neighbor // diameter) * diameter - radius
        spatial_distance_sq = dx * dx + dy * dy
        in_circle = spatial_distance_sq <= radius * radius

        ny = y[:, None] + dy[None, :]
        nx = x[:, None] + dx[None, :]
        # cv2.bilateralFilter uses BORDER_REFLECT_101 by default.
        ny = tl.where(ny < 0, -ny, tl.where(ny >= height, 2 * height - ny - 2, ny))
        nx = tl.where(nx < 0, -nx, tl.where(nx >= width, 2 * width - nx - 2, nx))
        neighbor_base = (ny * width + nx) * 3
        valid = valid_pixel[:, None] & (neighbor[None, :] < window) & in_circle[None, :]

        value_r = tl.load(source + neighbor_base, mask=valid, other=0.0).to(tl.float32)
        value_g = tl.load(source + neighbor_base + 1, mask=valid, other=0.0).to(tl.float32)
        value_b = tl.load(source + neighbor_base + 2, mask=valid, other=0.0).to(tl.float32)

        # Match OpenCV's 8-bit implementation, whose color lookup table is
        # indexed by the L1 distance across channels.
        color_distance = (
            tl.abs(value_r - center_r[:, None])
            + tl.abs(value_g - center_g[:, None])
            + tl.abs(value_b - center_b[:, None])
        )
        weight = tl.exp(
            -(color_distance * color_distance) / sigma_color_sq_x2
            - spatial_distance_sq[None, :].to(tl.float32) / sigma_space_sq_x2
        )
        weight = tl.where(valid, weight, 0.0)
        sum_r += tl.sum(weight * value_r, axis=1)
        sum_g += tl.sum(weight * value_g, axis=1)
        sum_b += tl.sum(weight * value_b, axis=1)
        sum_weight += tl.sum(weight, axis=1)

    out_r = tl.minimum(tl.maximum(sum_r / sum_weight + 0.5, 0.0), 255.0).to(tl.uint8)
    out_g = tl.minimum(tl.maximum(sum_g / sum_weight + 0.5, 0.0), 255.0).to(tl.uint8)
    out_b = tl.minimum(tl.maximum(sum_b / sum_weight + 0.5, 0.0), 255.0).to(tl.uint8)
    tl.store(destination + base, out_r, mask=valid_pixel)
    tl.store(destination + base + 1, out_g, mask=valid_pixel)
    tl.store(destination + base + 2, out_b, mask=valid_pixel)


class BilateralGaussian:
    """Fast CUDA bilateral filter for uint8 HWC RGB images."""

    def __call__(
        self,
        image_tensor: torch.Tensor,
        diameter: int = 30,
        sigma_color: float = 150.0,
        sigma_space: float = 100.0,
    ) -> torch.Tensor:
        if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB image, got shape {tuple(image_tensor.shape)}")
        if diameter <= 0 or sigma_color <= 0 or sigma_space <= 0:
            raise ValueError("diameter and sigma values must be positive")

        if image_tensor.dtype != torch.uint8:
            image_tensor = (image_tensor * 255).to(torch.uint8)
        image_tensor = image_tensor.to(device="cuda", non_blocking=True).contiguous()

        height, width, _ = image_tensor.shape
        output = torch.empty_like(image_tensor)
        radius = diameter // 2
        block_pixels = 32
        _bilateral_u8_hwc_kernel[(triton.cdiv(height * width, block_pixels),)](
            image_tensor,
            output,
            height=height,
            width=width,
            radius=radius,
            sigma_color_sq_x2=2.0 * sigma_color * sigma_color,
            sigma_space_sq_x2=2.0 * sigma_space * sigma_space,
            BLOCK_PIXELS=block_pixels,
            BLOCK_NEIGHBORS=32,
            num_warps=4,
        )
        return output
