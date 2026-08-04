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

"""Build an instanced USD scene for batched OVRTX rendering."""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mujoco_warp.ovrtx._config import TileLayout


@dataclasses.dataclass(frozen=True, slots=True)
class SceneLayout:
  """Paths and tensor ordering shared by the USD scene and runtime bridge."""

  scene_path: Path
  tile_layout: TileLayout
  environment_paths: tuple[str, ...]
  environment_offsets: tuple[tuple[float, float, float], ...]
  dynamic_geom_ids: tuple[int, ...]
  camera_ids: tuple[int, ...]
  update_camera_ids: tuple[int, ...]
  camera_transform_paths: tuple[str, ...]
  point_instancer_path: str | None
  product_paths: Mapping[int, str]
  render_vars: tuple[str, ...]

  @property
  def instance_count(self) -> int:
    return self.tile_layout.world_count * len(self.dynamic_geom_ids)


def dynamic_body_ids(model: mujoco.MjModel) -> tuple[int, ...]:
  """Return bodies whose world pose can change as the simulation advances."""
  dynamic = []
  for body_id in range(1, model.nbody):
    ancestor_id = body_id
    while ancestor_id:
      if model.body_jntnum[ancestor_id] or model.body_mocapid[ancestor_id] >= 0:
        dynamic.append(body_id)
        break
      ancestor_id = int(model.body_parentid[ancestor_id])
  return tuple(dynamic)


def dynamic_camera_ids(
  model: mujoco.MjModel,
  camera_ids: Sequence[int],
  body_ids: Sequence[int],
) -> tuple[int, ...]:
  """Return selected cameras whose world pose can change."""
  dynamic_bodies = set(body_ids)
  fixed_mode = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
  return tuple(
    camera_id
    for camera_id in camera_ids
    if int(model.cam_mode[camera_id]) != fixed_mode or int(model.cam_bodyid[camera_id]) in dynamic_bodies
  )


def _require_usd():
  try:
    from mujoco.usd import exporter
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdLux
    from pxr import UsdShade
    from pxr import Vt
  except ImportError as error:
    raise ImportError(
      "The OVRTX scene builder requires OpenUSD. Install the optional dependencies with `uv sync --extra ovrtx`."
    ) from error
  return exporter, Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt


def _usd_matrix(gf: Any, position: np.ndarray, rotation: np.ndarray):
  matrix = np.eye(4, dtype=np.float64)
  matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
  matrix[:3, 3] = np.asarray(position, dtype=np.float64)
  return gf.Matrix4d(matrix.T.tolist())


def _geom_local_rotation(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
  rotation = np.empty(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
  return rotation.reshape(3, 3)


def _set_transform(usd_geom: Any, gf: Any, prim: Any, position: np.ndarray, rotation: np.ndarray) -> None:
  xformable = usd_geom.Xformable(prim)
  for operation in xformable.GetOrderedXformOps():
    prim.RemoveProperty(operation.GetAttr().GetName())
  prim.RemoveProperty("xformOpOrder")
  xformable.AddTransformOp().Set(_usd_matrix(gf, position, rotation))


def _clear_transform(usd_geom: Any, prim: Any) -> None:
  xformable = usd_geom.Xformable(prim)
  for operation in xformable.GetOrderedXformOps():
    prim.RemoveProperty(operation.GetAttr().GetName())
  prim.RemoveProperty("xformOpOrder")


def _apply_ovrtx_api(sdf: Any, prim: Any) -> None:
  schemas = sdf.TokenListOp()
  schemas.prependedItems = ["OmniRtxDebugSettingsAPI_1"]
  prim.SetMetadata("apiSchemas", schemas)


def _setting_type(sdf: Any, value: Any):
  if isinstance(value, bool):
    return sdf.ValueTypeNames.Bool
  if isinstance(value, int):
    return sdf.ValueTypeNames.Int
  if isinstance(value, float):
    return sdf.ValueTypeNames.Float
  if isinstance(value, str):
    return sdf.ValueTypeNames.Token
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    if len(value) == 2:
      return sdf.ValueTypeNames.Float2
    if len(value) == 3:
      return sdf.ValueTypeNames.Float3
    if len(value) == 4:
      return sdf.ValueTypeNames.Float4
  raise TypeError(f"unsupported OVRTX setting value {value!r}")


def _setting_value(gf: Any, value: Any):
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    if len(value) == 2:
      return gf.Vec2f(*value)
    if len(value) == 3:
      return gf.Vec3f(*value)
    if len(value) == 4:
      return gf.Vec4f(*value)
  return value


def _heightfield_geometry(model: mujoco.MjModel, hfield_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Triangulate one MuJoCo height field in its local frame."""
  row_count = int(model.hfield_nrow[hfield_id])
  column_count = int(model.hfield_ncol[hfield_id])
  address = int(model.hfield_adr[hfield_id])
  size = np.asarray(model.hfield_size[hfield_id], dtype=np.float32)
  elevation = np.asarray(
    model.hfield_data[address : address + row_count * column_count],
    dtype=np.float32,
  ).reshape(row_count, column_count)

  x = np.linspace(-size[0], size[0], column_count, dtype=np.float32)
  y = np.linspace(-size[1], size[1], row_count, dtype=np.float32)
  grid_x, grid_y = np.meshgrid(x, y)
  points = np.stack((grid_x, grid_y, elevation * size[2]), axis=-1).reshape(-1, 3)

  row = np.arange(row_count - 1, dtype=np.int32)[:, None]
  column = np.arange(column_count - 1, dtype=np.int32)[None, :]
  lower_left = row * column_count + column
  lower_right = lower_left + 1
  upper_left = lower_left + column_count
  upper_right = upper_left + 1
  faces = np.stack(
    (
      lower_left,
      lower_right,
      upper_right,
      lower_left,
      upper_right,
      upper_left,
    ),
    axis=-1,
  ).reshape(-1, 3)

  u = np.linspace(0.0, 1.0, column_count, dtype=np.float32)
  v = np.linspace(0.0, 1.0, row_count, dtype=np.float32)
  grid_u, grid_v = np.meshgrid(u, v)
  texture_coordinates = np.stack((grid_u, grid_v), axis=-1).reshape(-1, 2)
  return points, faces, texture_coordinates


class _HeightFieldReference:
  """Small USD-exporter-compatible wrapper for an unsupported geom type."""

  def __init__(
    self,
    stage: Any,
    model: mujoco.MjModel,
    geom: Any,
    obj_name: str,
    gf: Any,
    sdf: Any,
    usd_geom: Any,
    usd_shade: Any,
    vt: Any,
    texture_file: str | None,
  ):
    self.geom = geom
    self.xform_path = f"/World/Mesh_Xform_{obj_name}"
    self._gf = gf
    self._usd_geom = usd_geom
    self._xform = usd_geom.Xform.Define(stage, self.xform_path)
    self._transform_op = self._xform.AddTransformOp()

    mesh = usd_geom.Mesh.Define(stage, f"{self.xform_path}/Mesh_{obj_name}")
    points, faces, texture_coordinates = _heightfield_geometry(model, int(model.geom_dataid[geom.objid]))
    mesh.GetPointsAttr().Set(vt.Vec3fArray.FromNumpy(points))
    mesh.GetFaceVertexCountsAttr().Set(vt.IntArray([3] * len(faces)))
    mesh.GetFaceVertexIndicesAttr().Set(vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.GetSubdivisionSchemeAttr().Set(usd_geom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)

    if geom.matid >= 0:
      texture_repeat = np.asarray(model.mat_texrepeat[geom.matid], dtype=np.float32)
      texture_coordinates *= texture_repeat
    primvar = usd_geom.PrimvarsAPI(mesh).CreatePrimvar(
      "UVMap",
      sdf.ValueTypeNames.TexCoord2fArray,
      usd_geom.Tokens.vertex,
    )
    primvar.Set(vt.Vec2fArray.FromNumpy(texture_coordinates))

    material_path = sdf.Path(f"/World/_materials/Material_{obj_name}")
    material = usd_shade.Material.Define(stage, material_path)
    surface = usd_shade.Shader.Define(stage, material_path.AppendPath("Principled_BSDF"))
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("opacity", sdf.ValueTypeNames.Float).Set(float(geom.rgba[3]))
    surface.CreateInput("metallic", sdf.ValueTypeNames.Float).Set(float(geom.shininess))
    surface.CreateInput("roughness", sdf.ValueTypeNames.Float).Set(1.0 - float(geom.shininess))

    if texture_file is None:
      surface.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).Set(tuple(float(value) for value in geom.rgba[:3]))
    else:
      image = usd_shade.Shader.Define(stage, material_path.AppendPath("Image_Texture"))
      image.CreateIdAttr("UsdUVTexture")
      image.CreateInput("file", sdf.ValueTypeNames.Asset).Set(texture_file)
      image.CreateInput("sourceColorSpace", sdf.ValueTypeNames.Token).Set("sRGB")
      image.CreateInput("wrapS", sdf.ValueTypeNames.Token).Set("repeat")
      image.CreateInput("wrapT", sdf.ValueTypeNames.Token).Set("repeat")
      image.CreateOutput("rgb", sdf.ValueTypeNames.Float3)
      uvmap = usd_shade.Shader.Define(stage, material_path.AppendPath("uvmap"))
      uvmap.CreateIdAttr("UsdPrimvarReader_float2")
      uvmap.CreateInput("varname", sdf.ValueTypeNames.Token).Set("UVMap")
      uvmap.CreateOutput("result", sdf.ValueTypeNames.Float2)
      image.CreateInput("st", sdf.ValueTypeNames.Float2).ConnectToSource(uvmap.ConnectableAPI(), "result")
      surface.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    mesh.GetPrim().ApplyAPI(usd_shade.MaterialBindingAPI)
    usd_shade.MaterialBindingAPI(mesh).Bind(material)

  def update(
    self,
    pos: np.ndarray,
    mat: np.ndarray,
    visible: bool,
    frame: int,
  ) -> None:
    self._transform_op.Set(_usd_matrix(self._gf, pos, mat), frame)
    visibility = self._usd_geom.Tokens.inherited if visible else self._usd_geom.Tokens.invisible
    self._xform.GetVisibilityAttr().Set(visibility, frame)


def _install_heightfield_exporter(
  usd_exporter: Any,
  model: mujoco.MjModel,
  gf: Any,
  sdf: Any,
  usd_geom: Any,
  usd_shade: Any,
  vt: Any,
) -> None:
  """Extend MuJoCo's USD exporter with renderable height fields."""
  original_load_geom = usd_exporter._load_geom

  def load_geom(geom: Any) -> None:
    if int(geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM) or int(geom.type) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
      original_load_geom(geom)
      return

    obj_name = usd_exporter._get_geom_name(geom)
    texture_file = None
    if geom.matid >= 0:
      texture_id = int(model.mat_texid[geom.matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
      if texture_id >= 0:
        texture_file = usd_exporter.texture_files[texture_id]
    reference = _HeightFieldReference(
      usd_exporter.stage,
      model,
      geom,
      obj_name,
      gf,
      sdf,
      usd_geom,
      usd_shade,
      vt,
      texture_file,
    )
    usd_exporter.geom_names.add(obj_name)
    usd_exporter.geom_refs[obj_name] = reference

  usd_exporter._load_geom = load_geom


def _author_camera(
  usd_geom: Any,
  gf: Any,
  stage: Any,
  xform_path: str,
  camera_path: str,
  position: np.ndarray,
  rotation: np.ndarray,
  fovy: float,
  width: int,
  height: int,
  znear: float,
  zfar: float,
) -> None:
  xform = usd_geom.Xform.Define(stage, xform_path)
  _set_transform(usd_geom, gf, xform.GetPrim(), position, rotation)

  camera = usd_geom.Camera.Define(stage, camera_path)
  horizontal_aperture = 20.955
  vertical_aperture = horizontal_aperture * height / width
  focal_length = vertical_aperture / (2.0 * math.tan(math.radians(fovy) / 2.0))
  camera.CreateProjectionAttr().Set(usd_geom.Tokens.perspective)
  camera.CreateHorizontalApertureAttr().Set(horizontal_aperture)
  camera.CreateVerticalApertureAttr().Set(vertical_aperture)
  camera.CreateFocalLengthAttr().Set(focal_length)
  camera.CreateClippingRangeAttr().Set(gf.Vec2f(znear, zfar))
  camera.CreateFStopAttr().Set(0.0)


def _author_lights(
  usd_geom: Any,
  usd_lux: Any,
  gf: Any,
  sdf: Any,
  stage: Any,
  dome_intensity: float,
  dome_texture: str | os.PathLike[str] | None,
  distant_intensity: float,
  distant_angle: float,
  distant_rotation: tuple[float, float, float],
) -> None:
  usd_geom.Scope.Define(stage, "/World/Lights")

  if dome_intensity > 0.0 or dome_texture is not None:
    dome = usd_lux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(float(dome_intensity))
    dome.CreateColorAttr().Set(gf.Vec3f(1.0, 1.0, 1.0))
    if dome_texture is not None:
      dome.CreateTextureFileAttr().Set(sdf.AssetPath(str(Path(dome_texture).expanduser().resolve())))
      dome.CreateTextureFormatAttr().Set("latlong")

  if distant_intensity > 0.0:
    distant = usd_lux.DistantLight.Define(stage, "/World/Lights/Distant")
    distant.CreateIntensityAttr().Set(float(distant_intensity))
    distant.CreateAngleAttr().Set(float(distant_angle))
    distant.CreateColorAttr().Set(gf.Vec3f(1.0, 1.0, 1.0))
    usd_geom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(gf.Vec3f(*distant_rotation))


def _author_render_products(
  gf: Any,
  sdf: Any,
  vt: Any,
  stage: Any,
  camera_targets: Mapping[int, Sequence[str]],
  render_vars: tuple[str, ...],
  render_mode: str,
  tile_layout: TileLayout,
  device_id: int,
  render_settings: Mapping[str, Any],
) -> dict[int, str]:
  stage.DefinePrim("/Render", "Scope")
  product_paths = {}
  for camera_id, targets in camera_targets.items():
    product_path = f"/Render/Camera_{camera_id}"
    product_paths[camera_id] = product_path
    product = stage.DefinePrim(product_path, "RenderProduct")
    _apply_ovrtx_api(sdf, product)
    product.CreateRelationship("camera").SetTargets([sdf.Path(target) for target in targets])
    product.CreateAttribute("resolution", sdf.ValueTypeNames.Int2).Set(
      gf.Vec2i(tile_layout.atlas_width, tile_layout.atlas_height)
    )
    product.CreateAttribute("deviceIds", sdf.ValueTypeNames.UIntArray).Set(vt.UIntArray([device_id]))
    product.CreateAttribute("omni:rtx:rendermode", sdf.ValueTypeNames.Token).Set(render_mode)

    render_var_paths = []
    for render_var in render_vars:
      if not sdf.Path.IsValidIdentifier(render_var):
        raise ValueError(f"render variable {render_var!r} is not a valid USD identifier")
      render_var_path = f"{product_path}/{render_var}"
      render_var_prim = stage.DefinePrim(render_var_path, "RenderVar")
      render_var_prim.CreateAttribute("sourceName", sdf.ValueTypeNames.String).Set(render_var)
      render_var_paths.append(sdf.Path(render_var_path))
    product.CreateRelationship("orderedVars").SetTargets(render_var_paths)

    for attribute, value in render_settings.items():
      product.CreateAttribute(str(attribute), _setting_type(sdf, value)).Set(_setting_value(gf, value))

  return product_paths


def build_scene(
  model: mujoco.MjModel,
  world_count: int,
  camera_ids: tuple[int, ...],
  render_vars: tuple[str, ...],
  render_mode: str,
  tile_layout: TileLayout,
  output_root: str | os.PathLike[str],
  device_id: int,
  *,
  world_spacing: float | None,
  dome_light_intensity: float,
  dome_light_texture: str | os.PathLike[str] | None,
  distant_light_intensity: float,
  distant_light_angle: float,
  distant_light_rotation: tuple[float, float, float],
  render_settings: Mapping[str, Any],
) -> SceneLayout:
  """Export one MuJoCo environment and describe its native OVStage clones."""
  exporter_module, gf, sdf, usd_geom, usd_lux, usd_shade, vt = _require_usd()
  output_root = Path(output_root)
  output_root.mkdir(parents=True, exist_ok=True)

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body_ids = dynamic_body_ids(model)
  dynamic_body_set = set(body_ids)
  usd_exporter = exporter_module.USDExporter(
    model,
    width=max(1, min(tile_layout.tile_width, int(model.vis.global_.offwidth))),
    height=max(1, min(tile_layout.tile_height, int(model.vis.global_.offheight))),
    max_geom=max(10_000, model.ngeom * 2),
    output_directory="scene",
    output_directory_root=str(output_root),
    camera_names=None,
    verbose=False,
  )
  _install_heightfield_exporter(usd_exporter, model, gf, sdf, usd_geom, usd_shade, vt)
  usd_exporter.update_scene(data)
  stage = usd_exporter.stage
  stage.SetMetadata("metersPerUnit", 1.0)
  usd_geom.SetStageUpAxis(stage, usd_geom.Tokens.z)

  geom_refs = {}
  for geom_ref in usd_exporter.geom_refs.values():
    if int(geom_ref.geom.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM):
      geom_refs[int(geom_ref.geom.objid)] = geom_ref

  # The exporter applies MuJoCo's visual geom-group mask and can intentionally
  # omit collision-only or otherwise hidden geoms. Its output is the authority
  # for what OVRTX should render and stream.
  exported_geom_ids = tuple(sorted(geom_refs))
  dynamic_geom_ids = tuple(geom_id for geom_id in exported_geom_ids if int(model.geom_bodyid[geom_id]) in dynamic_body_set)
  dynamic_geom_set = set(dynamic_geom_ids)

  # An over ancestor keeps static reference sources out of Hydra traversal.
  # Dynamic prototypes live below their PointInstancer, which is the standard
  # USD layout that prevents them from being rendered as standalone geometry.
  static_prototype_root = "/World/StaticPrototypes"
  stage.OverridePrim(static_prototype_root)
  point_instancer_path = None
  instancer = None
  if dynamic_geom_ids:
    point_instancer_path = "/World/DynamicGeomInstances"
    instancer = usd_geom.PointInstancer.Define(stage, point_instancer_path)
    usd_geom.Scope.Define(stage, f"{point_instancer_path}/Prototypes")

  root_layer = stage.GetRootLayer()
  prototype_paths = {}
  for geom_id in exported_geom_ids:
    if geom_id in dynamic_geom_set:
      prototype_path = f"{point_instancer_path}/Prototypes/Geom_{geom_id}"
    else:
      prototype_path = f"{static_prototype_root}/Geom_{geom_id}"
    prototype_paths[geom_id] = prototype_path
    source_path = geom_refs[geom_id].xform_path
    if not sdf.CopySpec(root_layer, sdf.Path(source_path), root_layer, sdf.Path(prototype_path)):
      raise RuntimeError(f"failed to copy exported geom {geom_id} to its prototype")
    stage.RemovePrim(source_path)
    prototype = stage.GetPrimAtPath(prototype_path)
    _clear_transform(usd_geom, prototype)
    prototype.RemoveProperty("visibility")
    if model.geom_rgba[geom_id, 3] <= 0.0:
      usd_geom.Imageable(prototype).CreateVisibilityAttr().Set(usd_geom.Tokens.invisible)

  keep_world_children = {"_materials", "StaticPrototypes", "DynamicGeomInstances"}
  world = stage.GetPrimAtPath("/World")
  for child in list(world.GetChildren()):
    if child.GetName() not in keep_world_children:
      stage.RemovePrim(child.GetPath())

  body_positions = np.asarray(data.xpos)
  body_rotations = np.asarray(data.xmat).reshape(model.nbody, 3, 3)
  camera_positions = np.asarray(data.cam_xpos)
  camera_rotations = np.asarray(data.cam_xmat).reshape(model.ncam, 3, 3)
  update_camera_ids = dynamic_camera_ids(model, camera_ids, body_ids)
  camera_transform_paths = []
  camera_targets = {camera_id: [] for camera_id in camera_ids}

  spacing = world_spacing
  if spacing is None:
    # OVRTX renders every environment in one scene, so leave enough room that
    # neighboring worlds stay outside a camera's useful view volume.
    spacing = max(1.0, 8.0 * float(model.stat.extent))
  if spacing <= 0.0:
    raise ValueError(f"world_spacing must be positive, got {spacing}")

  environment_paths = []
  environment_offsets = []
  for world_id in range(world_count):
    row = world_id // tile_layout.columns
    column = world_id % tile_layout.columns
    offset_x = (column - 0.5 * (tile_layout.columns - 1)) * spacing
    offset_y = (row - 0.5 * (tile_layout.rows - 1)) * spacing
    environment_path = f"/World/Environments/Env_{world_id}"
    environment_paths.append(environment_path)
    environment_offsets.append((offset_x, offset_y, 0.0))
    for camera_id in camera_ids:
      camera_xform_path = f"{environment_path}/Cameras/Camera_{camera_id}_Xform"
      if camera_id in update_camera_ids:
        camera_transform_paths.append(camera_xform_path)
      camera_targets[camera_id].append(f"{camera_xform_path}/Camera_{camera_id}")

  znear = max(1.0e-4, float(model.vis.map.znear) * float(model.stat.extent))
  zfar = max(znear * 2.0, float(model.vis.map.zfar) * float(model.stat.extent))
  if world_count > 1:
    # The default spacing puts all model geometry well inside this limit while
    # excluding neighboring environments from primary camera rays.
    zfar = min(zfar, max(znear * 2.0, 0.5 * spacing))
  usd_geom.Scope.Define(stage, "/World/Environments")
  environment_path = environment_paths[0]
  environment = usd_geom.Xform.Define(stage, environment_path)
  _set_transform(usd_geom, gf, environment.GetPrim(), np.zeros(3), np.eye(3))
  usd_geom.Scope.Define(stage, f"{environment_path}/Bodies")
  usd_geom.Scope.Define(stage, f"{environment_path}/Cameras")

  for body_id in range(model.nbody):
    geom_adr = int(model.body_geomadr[body_id])
    geom_num = int(model.body_geomnum[body_id])
    static_geom_ids = [
      geom_id
      for geom_id in range(geom_adr, geom_adr + geom_num)
      if geom_id in prototype_paths and geom_id not in dynamic_geom_set
    ]
    if not static_geom_ids:
      continue
    body_path = f"{environment_path}/Bodies/Body_{body_id}"
    body = usd_geom.Xform.Define(stage, body_path)
    _set_transform(usd_geom, gf, body.GetPrim(), body_positions[body_id], body_rotations[body_id])

    for geom_id in static_geom_ids:
      geom_path = f"{body_path}/Geom_{geom_id}"
      geom = usd_geom.Xform.Define(stage, geom_path)
      geom.GetPrim().GetReferences().AddInternalReference(prototype_paths[geom_id])
      geom.GetPrim().SetInstanceable(True)
      _set_transform(
        usd_geom,
        gf,
        geom.GetPrim(),
        model.geom_pos[geom_id],
        _geom_local_rotation(model, geom_id),
      )

  for camera_id in camera_ids:
    camera_xform_path = f"{environment_path}/Cameras/Camera_{camera_id}_Xform"
    camera_path = f"{camera_xform_path}/Camera_{camera_id}"
    _author_camera(
      usd_geom,
      gf,
      stage,
      camera_xform_path,
      camera_path,
      camera_positions[camera_id],
      camera_rotations[camera_id],
      float(model.cam_fovy[camera_id]),
      tile_layout.tile_width,
      tile_layout.tile_height,
      znear,
      zfar,
    )

  if dynamic_geom_ids:
    instancer.CreatePrototypesRel().SetTargets([sdf.Path(prototype_paths[geom_id]) for geom_id in dynamic_geom_ids])

    geom_count = len(dynamic_geom_ids)
    proto_indices = np.tile(np.arange(geom_count, dtype=np.int32), world_count)
    positions = np.tile(np.asarray(data.geom_xpos)[list(dynamic_geom_ids)], (world_count, 1)).astype(np.float32)
    positions += np.repeat(np.asarray(environment_offsets, dtype=np.float32), geom_count, axis=0)

    orientations = np.empty((geom_count, 4), dtype=np.float32)
    quaternion = np.empty(4, dtype=np.float64)
    for index, geom_id in enumerate(dynamic_geom_ids):
      mujoco.mju_mat2Quat(quaternion, data.geom_xmat[geom_id])
      orientations[index] = quaternion[[1, 2, 3, 0]]
    orientations = np.tile(orientations, (world_count, 1))

    instancer.CreateProtoIndicesAttr().Set(vt.IntArray.FromNumpy(proto_indices))
    instancer.CreatePositionsAttr().Set(vt.Vec3fArray.FromNumpy(positions))
    instancer.CreateOrientationsfAttr().Set(vt.QuatfArray.FromNumpy(orientations))

  _author_lights(
    usd_geom,
    usd_lux,
    gf,
    sdf,
    stage,
    dome_light_intensity,
    dome_light_texture,
    distant_light_intensity,
    distant_light_angle,
    distant_light_rotation,
  )
  product_paths = _author_render_products(
    gf,
    sdf,
    vt,
    stage,
    camera_targets,
    render_vars,
    render_mode,
    tile_layout,
    device_id,
    render_settings,
  )

  scene_path = output_root / "scene" / "frames" / "ovrtx_scene.usdc"
  scene_path.parent.mkdir(parents=True, exist_ok=True)
  if not root_layer.Export(str(scene_path)):
    raise RuntimeError(f"failed to export the OVRTX scene to {scene_path}")

  return SceneLayout(
    scene_path=scene_path,
    tile_layout=tile_layout,
    environment_paths=tuple(environment_paths),
    environment_offsets=tuple(environment_offsets),
    dynamic_geom_ids=dynamic_geom_ids,
    camera_ids=camera_ids,
    update_camera_ids=update_camera_ids,
    camera_transform_paths=tuple(camera_transform_paths),
    point_instancer_path=point_instancer_path,
    product_paths=product_paths,
    render_vars=render_vars,
  )


def populate_scene(stage: Any, layout: SceneLayout, ovstage: Any, ordinal: int = 1) -> int:
  """Populate one USD environment and clone it natively inside OVStage."""
  ovstage.population.open_usd(stage, str(layout.scene_path), ordinal=ordinal)
  stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()

  if len(layout.environment_paths) == 1:
    return ordinal

  ordinal += 1
  stage.clone(layout.environment_paths[0], layout.environment_paths[1:], ordinal=ordinal)
  stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()

  matrices = np.broadcast_to(np.eye(4, dtype=np.float64), (len(layout.environment_paths), 4, 4)).copy()
  matrices[:, 3, :3] = np.asarray(layout.environment_offsets)
  matrix_tensor = ovstage.make_dltensor(
    matrices,
    dtype=ovstage.numpy_to_dldatatype(matrices.dtype, lanes=16),
    shape=[len(layout.environment_paths)],
    ndim=1,
  )

  ordinal += 1
  with ovstage.PathDictionary(stage) as paths:
    path_list = paths.create_path_list_from_strings(layout.environment_paths)
    try:
      with stage.query_from_path_list(path_list) as query:
        stage.write_attribute(
          query,
          paths.intern_token("omni:xform"),
          ordinal=ordinal,
          tensors=matrix_tensor,
          is_array=False,
          semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()
    finally:
      paths.destroy_path_list(path_list)
  stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()

  # The USD layer contains only Env_0 when it is populated. Camera targets for
  # later environments are therefore dangling until the native clones above
  # exist. Rewrite each RenderProduct relationship after cloning so OVRTX
  # discovers every tiled view.
  ordinal += 1
  with ovstage.PathDictionary(stage) as paths:
    camera_attribute = paths.intern_token("camera")
    for camera_id, product_path in layout.product_paths.items():
      path_list = paths.create_path_list_from_strings([product_path])
      try:
        with stage.query_from_path_list(path_list) as query:
          camera_paths = np.asarray(
            [
              paths.intern_path(f"{environment_path}/Cameras/Camera_{camera_id}_Xform/Camera_{camera_id}")
              for environment_path in layout.environment_paths
            ],
            dtype=np.uint64,
          )
          stage.write_attribute(
            query,
            camera_attribute,
            ordinal=ordinal,
            tensors=camera_paths,
            is_array=True,
            semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
          ).wait()
      finally:
        paths.destroy_path_list(path_list)
  stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
  return ordinal
