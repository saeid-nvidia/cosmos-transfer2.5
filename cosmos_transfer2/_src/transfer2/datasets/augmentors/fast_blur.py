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

        # Import Triton lazily. PyTriton serializes Transfer2 callables while
        # starting the server; importing the JIT at module scope makes that
        # handshake unnecessarily expensive for every model instance.
        from cosmos_transfer2._src.transfer2.datasets.augmentors.triton_blur import bilateral_filter

        return bilateral_filter(image_tensor, diameter, sigma_color, sigma_space)
