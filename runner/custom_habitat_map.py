#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import scipy.ndimage

from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import utils

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass

cv2 = try_cv2_import()



def draw_point(
    image: np.ndarray,
    map_pose: Tuple[int, int],
    radius: float = 0.01,
    color: Tuple[int, int, int] = (0, 0, 127),
) -> np.ndarray:
    r"""Return an image with the agent image composited onto it.
    Args:
        image: the image onto which to put the agent.
        lidar_coord: the image coordinates where to paste the lidar points.
        lidar_radius: lidar_radius
    Returns:
        The modified background image. This operation is in place.
    """

    point_size = max(3, int(2 * radius))
    cv2.circle(
        image,
        map_pose,
        radius=point_size,
        color=color,
        thickness=-1,
    )
    return image

