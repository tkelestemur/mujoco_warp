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

"""Show per-world table texture randomization with MuJoCo Warp and Viser.

The scene is composed from ``benchmarks/franka_emika_panda/panda.xml`` with a
table added under the Panda base.

Run from the repository root:

  uv run python contrib/per_world_texture_randomization_viser.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANDA_XML = REPO_ROOT / "benchmarks/franka_emika_panda/panda.xml"
DEFAULT_PANDA_ASSET_DIR = REPO_ROOT / "benchmarks/mujoco_menagerie/franka_emika_panda/assets"
DEFAULT_TEXTURE_MANIFEST = REPO_ROOT / "contrib/assets/textures/table/manifest.json"

TABLE_HALF_EXTENTS = (0.60, 0.45)
TABLE_TOP_HALF_Z = 0.035
TABLE_TOP_CENTER_Z = -TABLE_TOP_HALF_Z
TABLE_TOP_Z = TABLE_TOP_CENTER_Z + TABLE_TOP_HALF_Z
TABLE_LEG_HALF_Z = 0.20
TABLE_LEG_CENTER_Z = TABLE_TOP_CENTER_Z - TABLE_TOP_HALF_Z - TABLE_LEG_HALF_Z

PANDA_CTRL_BASE = np.array([0.0, -0.55, 0.0, -1.85, 0.0, 1.70, 0.0, 0.02], dtype=np.float32)
PANDA_CTRL_AMP = np.array([0.45, 0.22, 0.38, 0.28, 0.35, 0.30, 0.55, 0.01], dtype=np.float32)
PANDA_CTRL_PHASES = np.array([0.0, 0.7, 1.6, 2.4, 3.1, 3.8, 4.5, 1.2], dtype=np.float32)
PANDA_INITIAL_QPOS = {
  "joint1": 0.0,
  "joint2": -0.55,
  "joint3": 0.0,
  "joint4": -1.85,
  "joint5": 0.0,
  "joint6": 1.70,
  "joint7": 0.0,
  "finger_joint1": 0.02,
  "finger_joint2": 0.02,
}


@dataclasses.dataclass(frozen=True)
class TextureSpec:
  name: str
  file: Path
  label: str


def _resolve_meshdir(panda_xml: Path, meshdir: str) -> Path:
  path = Path(meshdir)
  path = path if path.is_absolute() else panda_xml.parent / path
  if not path.exists() and DEFAULT_PANDA_ASSET_DIR.exists():
    path = DEFAULT_PANDA_ASSET_DIR
  return path.resolve()


def _load_texture_manifest(manifest_path: Path) -> tuple[TextureSpec, ...]:
  manifest_path = manifest_path.resolve()
  if not manifest_path.exists():
    raise FileNotFoundError(f"texture manifest not found: {manifest_path}")

  manifest = json.loads(manifest_path.read_text())
  texture_specs = []
  seen_names = set()
  for entry in manifest.get("textures", []):
    name = entry["name"]
    if name in seen_names:
      raise ValueError(f"duplicate texture name in {manifest_path}: {name!r}")

    file_path = manifest_path.parent / entry["file"]
    if not file_path.exists():
      raise FileNotFoundError(f"texture file listed in {manifest_path} does not exist: {file_path}")

    seen_names.add(name)
    texture_specs.append(TextureSpec(name=name, file=file_path.resolve(), label=entry.get("label", name)))

  if not texture_specs:
    raise ValueError(f"texture manifest does not contain any textures: {manifest_path}")
  return tuple(texture_specs)


def _add_demo_assets(spec: mujoco.MjSpec, texture_specs: Sequence[TextureSpec]) -> None:
  for tex in texture_specs:
    spec.add_texture(
      name=tex.name,
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      file=tex.file.as_posix(),
    )

  spec.add_material(
    name="table_mat",
    textures=["", texture_specs[0].name],
    texrepeat=[7, 5],
    texuniform=False,
    rgba=[1, 1, 1, 1],
  )
  spec.add_material(name="table_leg_mat", rgba=[0.25, 0.27, 0.29, 1])
  spec.add_material(name="floor_mat", rgba=[0.42, 0.43, 0.44, 1])


def _add_demo_worldbody(spec: mujoco.MjSpec) -> None:
  worldbody = spec.worldbody
  worldbody.add_camera(name="demo", pos=[0, -1.65, 0.95], xyaxes=[1, 0, 0, 0, 0.54, 0.84], fovy=46)
  worldbody.add_light(name="demo_key", pos=[-0.5, -1.0, 2.0], dir=[0.3, 0.5, -1], diffuse=[0.9, 0.9, 0.85])
  worldbody.add_light(name="demo_fill", pos=[1.2, 0.8, 1.2], dir=[-0.6, -0.3, -1], diffuse=[0.35, 0.40, 0.45])
  worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=[0, 0, -0.44],
    size=[3, 3, 0.02],
    material="floor_mat",
    contype=0,
    conaffinity=0,
  )

  table = worldbody.add_body(name="table", mocap=True)
  table.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[0, 0, TABLE_TOP_CENTER_Z],
    size=[TABLE_HALF_EXTENTS[0], TABLE_HALF_EXTENTS[1], TABLE_TOP_HALF_Z],
    material="table_mat",
    group=1,
    contype=0,
    conaffinity=0,
  )
  for name, x, y in (
    ("leg_fl", 0.48, 0.34),
    ("leg_fr", 0.48, -0.34),
    ("leg_bl", -0.48, 0.34),
    ("leg_br", -0.48, -0.34),
  ):
    table.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[x, y, TABLE_LEG_CENTER_Z],
      size=[0.035, 0.035, TABLE_LEG_HALF_Z],
      material="table_leg_mat",
      contype=0,
      conaffinity=0,
    )


def _build_demo_spec(panda_xml: Path, texture_specs: Sequence[TextureSpec]) -> mujoco.MjSpec:
  panda_xml = panda_xml.resolve()
  if not panda_xml.exists():
    raise FileNotFoundError(f"panda XML not found: {panda_xml}")

  panda = mujoco.MjSpec.from_file(panda_xml.as_posix())
  meshdir = _resolve_meshdir(panda_xml, panda.meshdir)
  panda.meshdir = meshdir.as_posix()

  spec = mujoco.MjSpec()
  spec.modelname = "per_world_texture_randomization"
  spec.meshdir = meshdir.as_posix()
  spec.option.timestep = 0.01
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.visual.headlight.active = 0
  spec.visual.quality.shadowsize = 2048

  panda_anchor = spec.worldbody.add_body(name="panda_anchor", mocap=True)
  panda_frame = panda_anchor.add_frame(name="panda_mount")
  spec.attach(panda, frame=panda_frame, prefix="")

  _add_demo_assets(spec, texture_specs)
  _add_demo_worldbody(spec)
  return spec


def _load_model(panda_xml: Path, texture_specs: Sequence[TextureSpec]) -> mujoco.MjModel:
  return _build_demo_spec(panda_xml, texture_specs).compile()


def _set_initial_state(mjm: mujoco.MjModel, mjd: mujoco.MjData) -> None:
  for joint_name, value in PANDA_INITIAL_QPOS.items():
    joint_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
      raise ValueError(f"joint {joint_name!r} was not found in the Panda model")
    mjd.qpos[mjm.jnt_qposadr[joint_id]] = value

  mjd.ctrl[:] = PANDA_CTRL_BASE
  mjm.qpos0[:] = mjd.qpos
  mujoco.mj_forward(mjm, mjd)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=4, help="number of parallel worlds")
  parser.add_argument("--randomize-every", type=int, default=50, help="texture reset interval in physics steps")
  parser.add_argument("--width", type=int, default=192, help="Warp render width per environment")
  parser.add_argument("--height", type=int, default=144, help="Warp render height per environment")
  parser.add_argument("--render-every", type=int, default=1, help="render every N Viser frames")
  parser.add_argument("--seed", type=int, default=0, help="random seed for texture choices")
  parser.add_argument("--device", default=None, help="Warp device, for example cuda:0 or cpu")
  parser.add_argument("--port", type=int, default=8080, help="Viser server port")
  parser.add_argument("--panda-xml", type=Path, default=DEFAULT_PANDA_XML, help="path to franka_emika_panda/panda.xml")
  parser.add_argument("--texture-manifest", type=Path, default=DEFAULT_TEXTURE_MANIFEST, help="offline texture manifest")
  return parser.parse_args(argv)


def _env_offsets(num_envs: int) -> np.ndarray:
  cols = int(math.ceil(math.sqrt(num_envs)))
  rows = int(math.ceil(num_envs / cols))
  spacing_x = 1.75
  spacing_y = 1.55
  offsets = np.zeros((num_envs, 3), dtype=np.float32)
  for world in range(num_envs):
    row, col = divmod(world, cols)
    offsets[world, 0] = (col - (cols - 1) * 0.5) * spacing_x
    offsets[world, 1] = ((rows - 1) * 0.5 - row) * spacing_y
  return offsets


def _make_texture_preview(mjm: mujoco.MjModel, texture_id: int) -> np.ndarray:
  width = int(mjm.tex_width[texture_id])
  height = int(mjm.tex_height[texture_id])
  nchannel = int(mjm.tex_nchannel[texture_id])
  adr = int(mjm.tex_adr[texture_id])
  if width < 1 or height < 1 or nchannel < 1 or adr < 0:
    raise ValueError(f"texture id {texture_id} has invalid compiled texture data")

  image = mjm.tex_data[adr : adr + width * height * nchannel].reshape(height, width, nchannel)
  if nchannel == 1:
    return np.repeat(image, 3, axis=2).astype(np.uint8)
  return image[:, :, :3].copy().astype(np.uint8)


def _unpack_rgb(packed_row: np.ndarray, width: int, height: int) -> np.ndarray:
  packed = packed_row.reshape(height, width).astype(np.uint32)
  b = (packed & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  return np.dstack([r, g, b])


def _patch_mjviser_compat() -> None:
  """Allow older MuJoCo builds to use the current mjviser GUI."""
  if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    setattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD", 0)


def _control_targets(step: int, num_envs: int, ctrlrange: np.ndarray) -> np.ndarray:
  targets = np.empty((num_envs, PANDA_CTRL_BASE.size), dtype=np.float32)
  for world in range(num_envs):
    phase = step * 0.025 + world * 0.55
    targets[world] = PANDA_CTRL_BASE + PANDA_CTRL_AMP * np.sin(phase + PANDA_CTRL_PHASES)
  return np.clip(targets, ctrlrange[:, 0], ctrlrange[:, 1]).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.num_envs < 1:
    raise ValueError(f"--num-envs must be positive, got {args.num_envs}.")
  if args.randomize_every < 1:
    raise ValueError(f"--randomize-every must be positive, got {args.randomize_every}.")
  if args.render_every < 1:
    raise ValueError(f"--render-every must be positive, got {args.render_every}.")

  try:
    import viser
    from mjviser import Viewer as MjViserViewer
  except ImportError as exc:
    raise RuntimeError("This demo requires mjviser and viser in the active environment.") from exc
  _patch_mjviser_compat()

  wp.config.quiet = True
  wp.init()
  if args.device is not None:
    wp.set_device(args.device)

  texture_specs = _load_texture_manifest(args.texture_manifest)
  mjm = _load_model(args.panda_xml, texture_specs)
  mjd = mujoco.MjData(mjm)
  _set_initial_state(mjm, mjd)

  num_envs = int(args.num_envs)
  rng = np.random.default_rng(args.seed)
  env_offsets = _env_offsets(num_envs)
  ctrlrange = mjm.actuator_ctrlrange.astype(np.float32)

  with wp.ScopedDevice(args.device):
    m = mjw.put_model(mjm, batch_sizes={"mat_texid": num_envs})
    d = mjw.put_data(mjm, mjd, nworld=num_envs)
    mjw.forward(m, d)

    rc = mjw.create_render_context(
      mjm,
      nworld=num_envs,
      cam_res=(args.width, args.height),
      render_rgb=True,
      render_depth=False,
      render_seg=False,
      use_textures=True,
      use_shadows=True,
      enabled_geom_groups=[0, 1, 2],
      enable_specular=True,
      enable_emission=False,
    )

    camera_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "demo")
    table_top_geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    table_mat_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
    texture_ids = np.array(
      [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_TEXTURE, spec.name) for spec in texture_specs], dtype=np.int32
    )
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    mat_texid = m.mat_texid.numpy()
    selected_texture = np.zeros(num_envs, dtype=np.int32)
    texture_previews = tuple(_make_texture_preview(mjm, int(texture_id)) for texture_id in texture_ids)

    table_image_handles: list[viser.ImageHandle] = []
    render_image_handles: list[viser.ImageHandle] = []
    label_handles: list[viser.LabelHandle] = []
    step_count = 0
    render_count = 0

    def set_table_textures() -> None:
      nonlocal selected_texture
      if len(texture_ids) >= num_envs:
        selected_texture = rng.choice(len(texture_ids), size=num_envs, replace=False).astype(np.int32)
      else:
        selected_texture = rng.integers(0, len(texture_ids), size=num_envs, dtype=np.int32)

      mat_texid[:, table_mat_id, rgb_role] = texture_ids[selected_texture]
      m.mat_texid.assign(mat_texid)

      for world, handle in enumerate(table_image_handles):
        handle.image = texture_previews[int(selected_texture[world])]
      for world, handle in enumerate(label_handles):
        handle.text = f"env {world} | {texture_specs[int(selected_texture[world])].label}"

    def _scene_offset(scene) -> np.ndarray:
      return np.asarray(getattr(scene, "_scene_offset", np.zeros(3)), dtype=np.float32)

    def table_surface_positions(scene) -> np.ndarray:
      positions = d.geom_xpos.numpy()[:, table_top_geom_id, :] + env_offsets + _scene_offset(scene)
      positions[:, 2] += TABLE_TOP_HALF_Z + 0.006
      return positions.astype(np.float32)

    def update_custom_visual_positions(scene) -> None:
      scene_offset = _scene_offset(scene)
      positions = table_surface_positions(scene)
      for world, handle in enumerate(table_image_handles):
        handle.position = positions[world]
      for world, handle in enumerate(render_image_handles):
        handle.position = env_offsets[world] + scene_offset + np.array([0.0, -1.00, 0.72], dtype=np.float32)
      for world, handle in enumerate(label_handles):
        handle.position = env_offsets[world] + scene_offset + np.array([-0.55, -0.78, 0.52], dtype=np.float32)

    def ensure_viser_handles(scene) -> None:
      if table_image_handles:
        return

      blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
      scene_offset = _scene_offset(scene)
      table_positions = table_surface_positions(scene)
      table_w = TABLE_HALF_EXTENTS[0] * 2.0
      table_h = TABLE_HALF_EXTENTS[1] * 2.0
      panel_w = 0.62
      panel_h = panel_w * args.height / args.width
      panel_wxyz = (0.70710678, 0.70710678, 0.0, 0.0)

      with scene.server.atomic():
        for world, offset in enumerate(env_offsets):
          table_image_handles.append(
            scene.server.scene.add_image(
              f"/texture_demo/env_{world}/table_texture",
              texture_previews[int(selected_texture[world])],
              render_width=table_w,
              render_height=table_h,
              wxyz=(1.0, 0.0, 0.0, 0.0),
              position=table_positions[world],
              cast_shadow=False,
              receive_shadow=False,
            )
          )
          render_image_handles.append(
            scene.server.scene.add_image(
              f"/texture_demo/env_{world}/warp_render",
              blank,
              render_width=panel_w,
              render_height=panel_h,
              wxyz=panel_wxyz,
              position=offset + scene_offset + np.array([0.0, -1.00, 0.72], dtype=np.float32),
              cast_shadow=False,
              receive_shadow=False,
            )
          )
          label_handles.append(
            scene.server.scene.add_label(
              f"/texture_demo/env_{world}/label",
              f"env {world} | {texture_specs[int(selected_texture[world])].label}",
              position=offset + scene_offset + np.array([-0.55, -0.78, 0.52], dtype=np.float32),
              font_size_mode="scene",
              font_scene_height=0.055,
            )
          )

    def render_warp_images() -> None:
      mjw.refit_bvh(m, d, rc)
      mjw.render(m, d, rc)
      rgb_all = rc.rgb_data.numpy()
      rgb_adr = rc.rgb_adr.numpy()
      start = int(rgb_adr[camera_id])
      stop = start + args.width * args.height
      for world, handle in enumerate(render_image_handles):
        handle.image = _unpack_rgb(rgb_all[world, start:stop], args.width, args.height)

    def step_fn(_mjm, _mjd) -> None:
      nonlocal step_count
      if step_count % args.randomize_every == 0:
        set_table_textures()

      d.ctrl.assign(_control_targets(step_count, num_envs, ctrlrange))
      mjw.step(m, d)
      mjw.get_data_into(_mjd, _mjm, d, world_id=0)
      step_count += 1

    def render_fn(scene) -> None:
      nonlocal render_count
      ensure_viser_handles(scene)

      body_xpos = d.xpos.numpy() + env_offsets[:, None, :]
      mocap_pos = d.mocap_pos.numpy() + env_offsets[:, None, :]
      scene.update_from_arrays(
        body_xpos=body_xpos,
        body_xmat=d.xmat.numpy(),
        mocap_pos=mocap_pos,
        mocap_quat=d.mocap_quat.numpy(),
        qpos=d.qpos.numpy(),
        qvel=d.qvel.numpy(),
        ctrl=d.ctrl.numpy(),
      )
      update_custom_visual_positions(scene)

      if render_count % args.render_every == 0:
        render_warp_images()
      render_count += 1

    server = viser.ViserServer(port=args.port)
    print(
      f"Running per-world texture demo at http://localhost:{args.port}\n"
      f"  worlds: {num_envs}\n"
      f"  texture reset interval: {args.randomize_every} physics steps\n"
      f"  Warp render resolution: {args.width}x{args.height}"
    )
    set_table_textures()
    MjViserViewer(mjm, mjd, step_fn=step_fn, render_fn=render_fn, num_envs=num_envs, server=server).run()


if __name__ == "__main__":
  main()
