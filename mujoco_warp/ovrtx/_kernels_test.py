# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import warp as wp
from absl.testing import absltest

from mujoco_warp.ovrtx._kernels import pack_render_state
from mujoco_warp.ovrtx._kernels import untile_rgba8


class KernelsTest(absltest.TestCase):
  def test_pack_render_state(self):
    geom_xpos = wp.array(
      np.array(
        [
          [[1, 2, 3], [4, 5, 6]],
          [[1, 2, 3], [14, 15, 16]],
        ],
        dtype=np.float32,
      ),
      dtype=wp.vec3,
      device="cpu",
    )
    geom_xmat = wp.array(
      np.array(
        [
          [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
          ],
          [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          ],
        ],
        dtype=np.float32,
      ),
      dtype=wp.mat33,
      device="cpu",
    )
    camera_xpos = wp.array(
      np.array([[[7, 8, 9]], [[17, 18, 19]]], dtype=np.float32),
      dtype=wp.vec3,
      device="cpu",
    )
    camera_xmat = wp.array(
      np.broadcast_to(np.eye(3, dtype=np.float32), (2, 1, 3, 3)),
      dtype=wp.mat33,
      device="cpu",
    )
    geom_ids = wp.array([1], dtype=wp.int32, device="cpu")
    camera_ids = wp.array([0], dtype=wp.int32, device="cpu")
    environment_offsets = wp.array([[-10, 0, 0], [10, 0, 0]], dtype=wp.vec3, device="cpu")
    positions = wp.empty(2, dtype=wp.vec3, device="cpu")
    orientations = wp.empty(2, dtype=wp.vec4, device="cpu")
    camera_transforms = wp.empty(2, dtype=wp.mat44d, device="cpu")

    pack_render_state(
      geom_xpos,
      geom_xmat,
      camera_xpos,
      camera_xmat,
      geom_ids,
      camera_ids,
      environment_offsets,
      positions,
      orientations,
      camera_transforms,
    )

    np.testing.assert_array_equal(positions.numpy(), [[-6, 5, 6], [24, 15, 16]])
    orientation_values = orientations.numpy()
    np.testing.assert_allclose(np.abs(np.dot(orientation_values[0], [0.0, 0.0, 2**-0.5, 2**-0.5])), 1.0)
    np.testing.assert_array_equal(orientation_values[1], [0, 0, 0, 1])
    np.testing.assert_array_equal(
      camera_transforms.numpy(),
      np.array(
        [
          [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [7, 8, 9, 1]],
          [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [17, 18, 19, 1]],
        ],
        dtype=np.float64,
      ),
    )

  def test_untile_rgba8(self):
    atlas_values = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    atlas = wp.array(atlas_values, dtype=wp.uint8, device="cpu")
    images = wp.empty((4, 2, 2, 4), dtype=wp.uint8, device="cpu")

    untile_rgba8(atlas, images, tile_width=2, tile_height=2, columns=2)

    expected = np.stack(
      [
        atlas_values[0:2, 0:2],
        atlas_values[0:2, 2:4],
        atlas_values[2:4, 0:2],
        atlas_values[2:4, 2:4],
      ]
    )
    np.testing.assert_array_equal(images.numpy(), expected)


if __name__ == "__main__":
  absltest.main()
