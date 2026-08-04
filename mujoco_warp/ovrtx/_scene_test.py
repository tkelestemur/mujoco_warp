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
import tempfile

import mujoco
from absl.testing import absltest

from mujoco_warp.ovrtx._config import choose_tile_layout
from mujoco_warp.ovrtx._scene import build_scene
from mujoco_warp.ovrtx._scene import dynamic_body_ids
from mujoco_warp.ovrtx._scene import dynamic_camera_ids
from mujoco_warp.ovrtx._scene import populate_scene

try:
  from pxr import Usd
  from pxr import UsdGeom

  _HAS_USD = True
except ImportError:
  _HAS_USD = False

_HAS_OVSTAGE = importlib.util.find_spec("ovstage") is not None


def _model():
  return mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <worldbody>
        <camera name="camera" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
        <geom type="plane" size="3 3 0.1"/>
        <body pos="0 0 1">
          <joint type="hinge"/>
          <geom type="box" size="0.2 0.3 0.4" rgba="0.8 0.2 0.1 1"/>
          <geom type="sphere" size="0.05" group="3"/>
          <body pos="0 0 0.5">
            <geom type="sphere" size="0.1"/>
          </body>
        </body>
        <body pos="2 0 0">
          <geom type="capsule" size="0.1 0.2"/>
        </body>
      </worldbody>
    </mujoco>
    """
  )


class SceneTest(absltest.TestCase):
  def test_dynamic_body_ids_include_welded_descendants(self):
    self.assertEqual(dynamic_body_ids(_model()), (1, 2))

  def test_static_camera_is_not_uploaded(self):
    model = _model()
    self.assertEqual(dynamic_camera_ids(model, (0,), dynamic_body_ids(model)), ())

    moving_camera_model = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <worldbody>
          <body>
            <joint/>
            <camera/>
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    body_ids = dynamic_body_ids(moving_camera_model)
    self.assertEqual(dynamic_camera_ids(moving_camera_model, (0,), body_ids), (0,))

  @absltest.skipUnless(_HAS_USD, "OpenUSD is not installed")
  def test_builds_instanced_batched_scene(self):
    model = _model()
    tile_layout = choose_tile_layout(3, 64, 48)
    with tempfile.TemporaryDirectory() as output_root:
      layout = build_scene(
        model,
        world_count=3,
        camera_ids=(0,),
        render_vars=("LdrColor",),
        render_mode="Minimal",
        tile_layout=tile_layout,
        output_root=output_root,
        device_id=0,
        world_spacing=None,
        dome_light_intensity=100.0,
        dome_light_texture=None,
        distant_light_intensity=1000.0,
        distant_light_angle=0.53,
        distant_light_rotation=(315.0, 0.0, 45.0),
        render_settings={"omni:rtx:minimal:castShadows": False},
      )

      stage = Usd.Stage.Open(str(layout.scene_path))
      self.assertEqual(stage.GetDefaultPrim().GetPath().pathString, "/World")
      self.assertEqual(layout.dynamic_geom_ids, (1, 3))
      self.assertEqual(layout.instance_count, 3 * 2)
      self.assertEmpty(layout.update_camera_ids)
      self.assertEmpty(layout.camera_transform_paths)
      self.assertEqual(layout.point_instancer_path, "/World/DynamicGeomInstances")

      self.assertEqual(
        layout.environment_paths,
        (
          "/World/Environments/Env_0",
          "/World/Environments/Env_1",
          "/World/Environments/Env_2",
        ),
      )
      expected_spacing = max(1.0, 8.0 * float(model.stat.extent))
      self.assertAlmostEqual(layout.environment_offsets[1][0] - layout.environment_offsets[0][0], expected_spacing)
      self.assertAlmostEqual(layout.environment_offsets[2][1] - layout.environment_offsets[0][1], expected_spacing)
      for geom_id in (0, 4):
        body_id = int(model.geom_bodyid[geom_id])
        instance = stage.GetPrimAtPath(f"/World/Environments/Env_0/Bodies/Body_{body_id}/Geom_{geom_id}")
        self.assertTrue(instance.IsValid())
        self.assertTrue(instance.IsInstanceable())
        self.assertTrue(instance.IsInstance())
      for geom_id in layout.dynamic_geom_ids:
        body_id = int(model.geom_bodyid[geom_id])
        instance = stage.GetPrimAtPath(f"/World/Environments/Env_0/Bodies/Body_{body_id}/Geom_{geom_id}")
        self.assertFalse(instance.IsValid())
      self.assertFalse(stage.GetPrimAtPath("/World/DynamicGeomInstances/Prototypes/Geom_2").IsValid())
      self.assertFalse(stage.GetPrimAtPath("/World/Environments/Env_1").IsValid())

      instancer = stage.GetPrimAtPath(layout.point_instancer_path)
      self.assertTrue(instancer.IsValid())
      self.assertEqual(instancer.GetAttribute("protoIndices").Get(), [0, 1, 0, 1, 0, 1])
      self.assertLen(instancer.GetAttribute("positions").Get(), layout.instance_count)
      self.assertLen(instancer.GetAttribute("orientationsf").Get(), layout.instance_count)
      self.assertEqual(
        [path.pathString for path in instancer.GetRelationship("prototypes").GetTargets()],
        [
          "/World/DynamicGeomInstances/Prototypes/Geom_1",
          "/World/DynamicGeomInstances/Prototypes/Geom_3",
        ],
      )

      product = stage.GetPrimAtPath("/Render/Camera_0")
      self.assertEqual(product.GetAttribute("resolution").Get(), (128, 96))
      self.assertLen(product.GetRelationship("camera").GetTargets(), 3)
      self.assertFalse(product.GetAttribute("omni:rtx:minimal:castShadows").Get())
      camera = stage.GetPrimAtPath("/World/Environments/Env_0/Cameras/Camera_0_Xform/Camera_0")
      self.assertAlmostEqual(camera.GetAttribute("clippingRange").Get()[1], 0.5 * expected_spacing)

      for geom_id in (0, 4):
        self.assertTrue(stage.GetPrimAtPath(f"/World/StaticPrototypes/Geom_{geom_id}").IsValid())
      for geom_id in layout.dynamic_geom_ids:
        prototype = stage.GetPrimAtPath(f"/World/DynamicGeomInstances/Prototypes/Geom_{geom_id}")
        self.assertTrue(prototype.IsValid())
        self.assertTrue(prototype.IsDefined())

  @absltest.skipUnless(_HAS_USD, "OpenUSD is not installed")
  def test_builds_heightfield_mesh(self):
    model = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <asset>
          <hfield name="terrain" nrow="3" ncol="3" size="1 1 .5 .1"
            elevation="0 0 0 0 1 0 0 0 0"/>
        </asset>
        <worldbody>
          <camera pos="0 -3 2" xyaxes="1 0 0 0 .55 1"/>
          <geom type="hfield" hfield="terrain" rgba=".2 .7 .25 1"/>
        </worldbody>
      </mujoco>
      """
    )
    with tempfile.TemporaryDirectory() as output_root:
      layout = build_scene(
        model,
        world_count=2,
        camera_ids=(0,),
        render_vars=("LdrColor",),
        render_mode="Minimal",
        tile_layout=choose_tile_layout(2, 32, 32),
        output_root=output_root,
        device_id=0,
        world_spacing=None,
        dome_light_intensity=100.0,
        dome_light_texture=None,
        distant_light_intensity=1000.0,
        distant_light_angle=0.53,
        distant_light_rotation=(315.0, 0.0, 45.0),
        render_settings={},
      )

      stage = Usd.Stage.Open(str(layout.scene_path))
      prototype = stage.GetPrimAtPath("/World/StaticPrototypes/Geom_0")
      self.assertTrue(prototype.IsValid())
      meshes = [child for child in prototype.GetAllChildren() if child.IsA(UsdGeom.Mesh)]
      self.assertLen(meshes, 1)
      mesh = UsdGeom.Mesh(meshes[0])
      self.assertLen(mesh.GetPointsAttr().Get(), 9)
      self.assertLen(mesh.GetFaceVertexIndicesAttr().Get(), 24)
      self.assertTrue(stage.GetPrimAtPath("/World/Environments/Env_0/Bodies/Body_0/Geom_0").IsValid())

  @absltest.skipUnless(_HAS_USD and _HAS_OVSTAGE, "OpenUSD and OVStage are not installed")
  def test_scene_populates_in_ovstage(self):
    import ovstage

    model = _model()
    with tempfile.TemporaryDirectory() as output_root:
      layout = build_scene(
        model,
        world_count=2,
        camera_ids=(0,),
        render_vars=("LdrColor",),
        render_mode="Minimal",
        tile_layout=choose_tile_layout(2, 32, 32),
        output_root=output_root,
        device_id=0,
        world_spacing=None,
        dome_light_intensity=0.0,
        dome_light_texture=None,
        distant_light_intensity=1.0,
        distant_light_angle=0.53,
        distant_light_rotation=(315.0, 0.0, 45.0),
        render_settings={},
      )

      with ovstage.Stage("mujoco_warp.ovrtx.test") as stage:
        ordinal = populate_scene(stage, layout, ovstage)
        self.assertEqual(ordinal, 4)
        with ovstage.PathDictionary(stage) as paths:
          camera_paths = tuple(f"{environment_path}/Cameras/Camera_0_Xform" for environment_path in layout.environment_paths)
          camera_path_list = paths.create_path_list_from_strings(camera_paths)
          instance_path_list = paths.create_path_list_from_strings([layout.point_instancer_path])
          try:
            with stage.query_from_path_list(camera_path_list) as query:
              attribute = paths.intern_token("omni:xform")
              count = 0
              with stage.read_attributes(query, [attribute], ovstage.OrdinalRange.latest(ordinal)) as read:
                read.wait()
                for group in read.groups():
                  count += group.prim_count
                  stage.release_group(group)
              self.assertEqual(count, len(camera_paths))
            with stage.query_from_path_list(instance_path_list) as query:
              attributes = [paths.intern_token("positions"), paths.intern_token("orientationsf")]
              count = 0
              with stage.read_attributes(query, attributes, ovstage.OrdinalRange.latest(ordinal)) as read:
                read.wait()
                for group in read.groups():
                  count += group.prim_count
                  stage.release_group(group)
              self.assertEqual(count, len(attributes))
          finally:
            paths.destroy_path_list(instance_path_list)
            paths.destroy_path_list(camera_path_list)


if __name__ == "__main__":
  absltest.main()
