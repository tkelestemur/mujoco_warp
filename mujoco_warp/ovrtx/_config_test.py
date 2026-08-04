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

from absl.testing import absltest

from mujoco_warp.ovrtx._config import RenderMode
from mujoco_warp.ovrtx._config import choose_tile_layout
from mujoco_warp.ovrtx._config import normalize_camera_ids
from mujoco_warp.ovrtx._config import normalize_render_mode


class ConfigTest(absltest.TestCase):
  def test_tile_layout(self):
    layout = choose_tile_layout(10, 64, 48)

    self.assertEqual(layout.columns, 4)
    self.assertEqual(layout.rows, 3)
    self.assertEqual(layout.atlas_width, 256)
    self.assertEqual(layout.atlas_height, 144)

  def test_explicit_tile_columns(self):
    layout = choose_tile_layout(10, 32, 32, columns=5)

    self.assertEqual((layout.columns, layout.rows), (5, 2))

  def test_camera_ids(self):
    self.assertEqual(normalize_camera_ids(None, 3), (0, 1, 2))
    self.assertEqual(normalize_camera_ids([2, 0], 3), (2, 0))
    with self.assertRaisesRegex(ValueError, "duplicates"):
      normalize_camera_ids([1, 1], 3)
    with self.assertRaisesRegex(ValueError, "outside"):
      normalize_camera_ids([3], 3)

  def test_render_mode(self):
    self.assertEqual(
      normalize_render_mode(RenderMode.REAL_TIME_PATH_TRACING),
      "RealTimePathTracing",
    )
    with self.assertRaisesRegex(ValueError, "unsupported"):
      normalize_render_mode("fast-ish")


if __name__ == "__main__":
  absltest.main()
