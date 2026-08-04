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

import importlib.util
import os

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp.ovrtx import Renderer
from mujoco_warp.ovrtx import RendererConfig
from mujoco_warp.ovrtx import RenderMode

_RUN_INTEGRATION_TEST = (
  os.environ.get("MUJOCO_WARP_OVRTX_TEST") == "1"
  and importlib.util.find_spec("ovrtx") is not None
  and importlib.util.find_spec("ovstage") is not None
  and wp.get_device().is_cuda
)


@absltest.skipUnless(_RUN_INTEGRATION_TEST, "set MUJOCO_WARP_OVRTX_TEST=1 on an OVRTX-capable CUDA host")
class RendererTest(absltest.TestCase):
  def test_render_cuda_batch(self):
    model = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <worldbody>
          <camera name="camera" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
          <camera name="camera_2" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
          <geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.5">
            <joint type="hinge"/>
            <geom type="box" size="0.2 0.3 0.4" rgba="0.8 0.2 0.1 1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    cpu_data = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu_data)
    data = mjw.put_data(model, cpu_data, nworld=2)
    config = RendererConfig(
      width=32,
      height=32,
      camera_ids=(0, 1),
      render_mode=RenderMode.MINIMAL,
    )

    with Renderer(model, world_count=2, config=config) as renderer:
      with renderer.render(data) as frame:
        pixels_by_camera = []
        for camera_id in (0, 1):
          with frame.map(camera_id=camera_id) as mapped:
            pixels_by_camera.append(mapped.as_batch().numpy())

    for pixels in pixels_by_camera:
      self.assertEqual(pixels.shape, (2, 32, 32, 4))
      self.assertGreater(np.count_nonzero(pixels[..., :3]), 0)
      # RTX sampling and denoising are not bit-identical across tiles, even for
      # identical worlds. Their mean pixel error should nevertheless stay small.
      difference = np.abs(pixels[0, ..., :3].astype(np.int16) - pixels[1, ..., :3].astype(np.int16))
      self.assertLess(float(np.mean(difference)), 10.0)

  def test_dynamic_geom_update_reaches_rtx(self):
    model = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <worldbody>
          <camera pos="0 -4 2" xyaxes="1 0 0 0 .45 1"/>
          <geom type="plane" size="3 3 .1"/>
          <body pos="-.7 0 .5">
            <freejoint/>
            <geom type="box" size=".45 .12 .3" rgba="1 .05 .02 1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    cpu_data = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu_data)
    data = mjw.put_data(model, cpu_data, nworld=1)
    config = RendererConfig(
      width=160,
      height=120,
      camera_ids=(0,),
      render_mode=RenderMode.REAL_TIME_PATH_TRACING,
    )

    with Renderer(model, world_count=1, config=config) as renderer:
      renderer.warmup(data, frames=2)
      with renderer.render(data) as frame:
        with frame.map() as mapped:
          before = mapped.as_batch().numpy()[0]

      positions = data.geom_xpos.numpy()
      positions[0, 1, 0] = 0.7
      wp.copy(data.geom_xpos, wp.array(positions, dtype=wp.vec3, device=data.geom_xpos.device))
      with renderer.render(data) as frame:
        with frame.map() as mapped:
          after_position = mapped.as_batch().numpy()[0]

      rotations = data.geom_xmat.numpy()
      rotations[0, 1] = np.array(
        [
          [0.0, -1.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
      )
      wp.copy(data.geom_xmat, wp.array(rotations, dtype=wp.mat33, device=data.geom_xmat.device))
      with renderer.render(data) as frame:
        with frame.map() as mapped:
          after_rotation = mapped.as_batch().numpy()[0]

    position_difference = np.abs(before[..., :3].astype(np.int16) - after_position[..., :3].astype(np.int16))
    rotation_difference = np.abs(after_position[..., :3].astype(np.int16) - after_rotation[..., :3].astype(np.int16))
    self.assertGreater(float(np.mean(position_difference)), 5.0)
    self.assertGreater(float(np.mean(rotation_difference)), 2.0)

  def test_dynamic_camera_update_reaches_rtx(self):
    model = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <worldbody>
          <geom type="plane" size="3 3 .1"/>
          <geom type="box" pos="0 0 .5" size=".25 .25 .5" rgba="1 .05 .02 1"/>
          <body pos="0 -4 2">
            <inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>
            <freejoint/>
            <camera pos="0 0 0" xyaxes="1 0 0 0 .45 1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    cpu_data = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu_data)
    data = mjw.put_data(model, cpu_data, nworld=2)
    config = RendererConfig(
      width=160,
      height=120,
      camera_ids=(0,),
      render_mode=RenderMode.REAL_TIME_PATH_TRACING,
    )

    with Renderer(model, world_count=2, config=config) as renderer:
      renderer.warmup(data, frames=2)
      with renderer.render(data) as frame:
        with frame.map() as mapped:
          before = mapped.as_batch().numpy()

      camera_positions = data.cam_xpos.numpy()
      camera_positions[1, 0, 0] = 1.2
      wp.copy(data.cam_xpos, wp.array(camera_positions, dtype=wp.vec3, device=data.cam_xpos.device))
      with renderer.render(data) as frame:
        with frame.map() as mapped:
          after = mapped.as_batch().numpy()

    difference = np.abs(before[..., :3].astype(np.int16) - after[..., :3].astype(np.int16))
    self.assertLess(float(np.mean(difference[0])), 5.0)
    self.assertGreater(float(np.mean(difference[1])), 5.0)

  def test_textured_heightfield_reaches_rtx(self):
    model = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <asset>
          <texture name="checker" type="2d" builtin="checker"
            rgb1=".05 .3 .02" rgb2=".5 .15 .02" width="64" height="64"/>
          <material name="terrain" texture="checker" texrepeat="4 4"/>
          <hfield name="terrain" nrow="3" ncol="3" size="1 1 .5 .1"
            elevation="0 0 0 0 1 0 0 0 0"/>
        </asset>
        <worldbody>
          <camera pos="0 -3 2" xyaxes="1 0 0 0 .55 1"/>
          <geom type="hfield" hfield="terrain" material="terrain"/>
        </worldbody>
      </mujoco>
      """
    )
    cpu_data = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu_data)
    data = mjw.put_data(model, cpu_data, nworld=2)
    config = RendererConfig(
      width=96,
      height=96,
      camera_ids=(0,),
      render_mode=RenderMode.REAL_TIME_PATH_TRACING,
    )

    with Renderer(model, world_count=2, config=config) as renderer:
      renderer.warmup(data, frames=1)
      with renderer.render(data) as frame:
        with frame.map() as mapped:
          pixels = mapped.as_batch().numpy()

    self.assertEqual(pixels.shape, (2, 96, 96, 4))
    self.assertGreater(float(np.std(pixels[..., :3])), 10.0)


if __name__ == "__main__":
  absltest.main()
