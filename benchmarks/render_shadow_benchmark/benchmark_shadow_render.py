"""Benchmark MJWarp render throughput with shadows disabled and enabled.

This script renders a simple table-top scene across many worlds. It is intended
for profiling the shadow path, not as a correctness test.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import warp as wp

import mujoco_warp as mjw

SCENE = Path(__file__).with_name("shadow_table_scene.xml")


FR3_SCENE = """<mujoco model="shadow_table_fr3_benchmark">
  <include file="{fr3_xml}"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <headlight active="0"/>
    <map znear="0.01" zfar="12"/>
    <quality shadowsize="2048"/>
    <global offwidth="512" offheight="512"/>
  </visual>

  <asset>
    <material name="bench_ground" rgba="0.42 0.43 0.41 1"/>
    <material name="bench_table" rgba="0.55 0.38 0.22 1"/>
    <material name="bench_red" rgba="0.85 0.20 0.16 1"/>
    <material name="bench_blue" rgba="0.12 0.32 0.86 1"/>
    <material name="bench_green" rgba="0.16 0.62 0.26 1"/>
    <material name="bench_yellow" rgba="0.95 0.78 0.18 1"/>
    <material name="bench_gray" rgba="0.55 0.57 0.60 1"/>
  </asset>

  <worldbody>
    <camera
      name="overview"
      pos="2.45 -2.85 1.75"
      xyaxes="0.759257 0.650791 0 -0.208913 0.243732 0.947074"
      fovy="45"
      resolution="120 160"
      output="rgb"/>

    <light
      name="left_overhead"
      pos="-1.0 -0.8 2.9"
      dir="0.35 0.25 -1"
      diffuse="0.82 0.78 0.70"
      specular="0.06 0.06 0.05"
      cutoff="65"
      castshadow="true"/>
    <light
      name="right_overhead"
      pos="1.0 0.7 2.6"
      dir="-0.35 -0.15 -1"
      diffuse="0.55 0.60 0.72"
      specular="0.04 0.05 0.06"
      cutoff="75"
      castshadow="true"/>
    <light
      name="rear_left_overhead"
      pos="-1.15 0.85 2.55"
      dir="0.35 -0.25 -1"
      diffuse="0.45 0.50 0.62"
      specular="0.04 0.04 0.05"
      cutoff="70"
      castshadow="true"/>
    <light
      name="front_right_overhead"
      pos="1.10 -0.90 2.75"
      dir="-0.40 0.25 -1"
      diffuse="0.62 0.56 0.48"
      specular="0.04 0.04 0.04"
      cutoff="70"
      castshadow="true"/>

    <geom name="ground" type="plane" pos="0 0 0" size="4 4 0.05" material="bench_ground"/>

    <geom name="table_top" type="box" pos="0 0 0.62" size="0.85 0.55 0.045" material="bench_table"/>
    <geom name="leg_front_left" type="cylinder" pos="-0.72 -0.42 0.31" size="0.035 0.31" material="bench_table"/>
    <geom name="leg_front_right" type="cylinder" pos="0.72 -0.42 0.31" size="0.035 0.31" material="bench_table"/>
    <geom name="leg_back_left" type="cylinder" pos="-0.72 0.42 0.31" size="0.035 0.31" material="bench_table"/>
    <geom name="leg_back_right" type="cylinder" pos="0.72 0.42 0.31" size="0.035 0.31" material="bench_table"/>

    <geom name="box_object" type="box" pos="0.10 -0.24 0.76" size="0.13 0.13 0.095" material="bench_red"/>
    <geom name="sphere_object" type="sphere" pos="0.43 -0.18 0.80" size="0.14" material="bench_blue"/>
    <geom name="capsule_object" type="capsule" pos="0.50 0.12 0.81" size="0.065 0.20" euler="0 1.309 0.489" material="bench_green"/>
    <geom name="cylinder_object" type="cylinder" pos="0.05 0.25 0.79" size="0.10 0.13" material="bench_yellow"/>
    <geom name="ellipsoid_object" type="ellipsoid" pos="-0.28 0.20 0.78" size="0.11 0.07 0.13" euler="0 0.349 -0.611" material="bench_gray"/>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class _Case:
  label: str
  use_shadows: bool
  light_castshadow: bool


@dataclass(frozen=True)
class _Timing:
  label: str
  elapsed_s: float
  iterations: int
  nworld: int
  width: int
  height: int

  @property
  def batch_fps(self) -> float:
    return self.iterations / self.elapsed_s

  @property
  def world_fps(self) -> float:
    return self.iterations * self.nworld / self.elapsed_s

  @property
  def megapixels_per_s(self) -> float:
    return self.iterations * self.nworld * self.width * self.height / self.elapsed_s / 1.0e6


CASES = {
  "shadows": _Case("shadows", use_shadows=True, light_castshadow=True),
  "no_light_shadows": _Case("no_light_shadows", use_shadows=True, light_castshadow=False),
  "no_shadows": _Case("no_shadows", use_shadows=False, light_castshadow=True),
}


def _fr3_dir_candidates() -> list[Path]:
  candidates = []
  env_dir = os.environ.get("MJW_FR3_DIR")
  if env_dir:
    candidates.append(Path(env_dir))
  candidates.extend(
    [
      Path(__file__).parents[1] / "mujoco_menagerie" / "franka_fr3",
      Path.home() / "code" / "mujoco_menagerie" / "franka_fr3",
      Path.home() / "code" / "eka-robotics" / "third_party" / "mujoco_menagerie" / "franka_fr3",
      Path.home() / "code" / "cursor-sync" / "eka-robotics-rendering" / "third_party" / "mujoco_menagerie" / "franka_fr3",
    ]
  )
  return candidates


def _find_fr3_dir(fr3_dir: str | None) -> Path:
  candidates = [Path(fr3_dir)] if fr3_dir else _fr3_dir_candidates()
  for candidate in candidates:
    if (candidate / "fr3.xml").is_file() and (candidate / "assets").is_dir():
      return candidate

  searched = "\n  ".join(path.as_posix() for path in candidates)
  raise FileNotFoundError(
    "Could not find MuJoCo Menagerie franka_fr3 assets. Pass --fr3-dir or set MJW_FR3_DIR. Searched:\n  "
    + searched
  )


def _make_fr3_scene(fr3_dir: Path, tmpdir: Path) -> Path:
  fr3_text = (fr3_dir / "fr3.xml").read_text()
  fr3_text = fr3_text.replace(
    '<compiler angle="radian" meshdir="assets"/>',
    f'<compiler angle="radian" meshdir="{(fr3_dir / "assets").as_posix()}"/>',
  )
  fr3_text = fr3_text.replace(
    '<body name="base" childclass="fr3">',
    '<body name="base" childclass="fr3" pos="-0.52 0 0.665" euler="0 0 -1.57079632679">',
    1,
  )

  fr3_xml = tmpdir / "fr3_shadow_benchmark.xml"
  fr3_xml.write_text(fr3_text)

  scene_xml = tmpdir / "shadow_table_fr3_scene.xml"
  scene_xml.write_text(FR3_SCENE.format(fr3_xml=fr3_xml.as_posix()))
  return scene_xml


def _parse_cases(value: str) -> list[_Case]:
  cases = []
  for name in value.split(","):
    name = name.strip()
    if not name:
      continue
    if name not in CASES:
      choices = ", ".join(sorted(CASES))
      raise argparse.ArgumentTypeError(f"case must be one of: {choices}")
    cases.append(CASES[name])
  if not cases:
    raise argparse.ArgumentTypeError("at least one case is required")
  return cases


def _load_model(light_castshadow: bool, scene: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
  mjm = mujoco.MjModel.from_xml_path(scene.as_posix())
  mjm.light_castshadow[:] = light_castshadow
  mjd = mujoco.MjData(mjm)
  if mjm.nkey:
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
  mujoco.mj_forward(mjm, mjd)
  return mjm, mjd


def _make_context(mjm: mujoco.MjModel, nworld: int, width: int, height: int, use_shadows: bool) -> mjw.RenderContext:
  return mjw.create_render_context(
    mjm,
    nworld=nworld,
    cam_res=(width, height),
    render_rgb=True,
    render_depth=False,
    render_seg=False,
    use_textures=False,
    use_shadows=use_shadows,
    use_ambient_lighting=True,
    render_skybox=False,
  )


def _run_iteration(m: mjw.Model, d: mjw.Data, rc: mjw.RenderContext, include_step: bool, include_refit: bool):
  if include_step:
    mjw.step(m, d)
  if include_refit:
    mjw.refit_bvh(m, d, rc)
  mjw.render(m, d, rc)


def _benchmark(
  label: str,
  m: mjw.Model,
  d: mjw.Data,
  rc: mjw.RenderContext,
  nworld: int,
  width: int,
  height: int,
  warmup: int,
  iterations: int,
  include_step: bool,
  include_refit: bool,
) -> _Timing:
  for _ in range(warmup):
    _run_iteration(m, d, rc, include_step, include_refit)
  wp.synchronize()

  start = time.perf_counter()
  for _ in range(iterations):
    _run_iteration(m, d, rc, include_step, include_refit)
  wp.synchronize()

  return _Timing(
    label=label,
    elapsed_s=time.perf_counter() - start,
    iterations=iterations,
    nworld=nworld,
    width=width,
    height=height,
  )


def _print_timing(timing: _Timing):
  print(
    f"{timing.label:>14} "
    f"elapsed={timing.elapsed_s:.4f}s "
    f"batch_fps={timing.batch_fps:.3f} "
    f"world_fps={timing.world_fps:.1f} "
    f"mpix_s={timing.megapixels_per_s:.1f}"
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--nworld", type=int, default=512)
  parser.add_argument("--width", type=int, default=120)
  parser.add_argument("--height", type=int, default=160)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument(
    "--scene",
    choices=("fr3", "table"),
    default="fr3",
    help="benchmark scene to render",
  )
  parser.add_argument("--fr3-dir", type=str, default=None, help="path to mujoco_menagerie/franka_fr3")
  parser.add_argument(
    "--cases",
    type=_parse_cases,
    default=_parse_cases("no_shadows,no_light_shadows,shadows"),
    help="comma-separated cases: no_shadows,no_light_shadows,shadows",
  )
  parser.add_argument(
    "--render-only",
    action="store_true",
    help="skip mjw.step and measure refit_bvh+render only",
  )
  parser.add_argument(
    "--skip-refit",
    action="store_true",
    help="skip refit_bvh and measure render only; valid for static benchmark scenes",
  )
  args = parser.parse_args()

  wp.init()
  with tempfile.TemporaryDirectory(prefix="mjw_shadow_bench_") as tmp, wp.ScopedDevice(args.device):
    scene = SCENE
    if args.scene == "fr3":
      fr3_dir = _find_fr3_dir(args.fr3_dir)
      scene = _make_fr3_scene(fr3_dir, Path(tmp))

    if args.skip_refit:
      mode = "render"
    elif args.render_only:
      mode = "refit+render"
    else:
      mode = "step+refit+render"
    print(f"Scene: {scene}")
    print(f"Device: {wp.get_device()}")
    print(
      f"Mode: {mode}, nworld={args.nworld}, resolution={args.width}x{args.height}, "
      f"iterations={args.iterations}, warmup={args.warmup}"
    )

    timings = []
    for case in args.cases:
      mjm, mjd = _load_model(case.light_castshadow, scene)
      m = mjw.put_model(mjm)
      d = mjw.put_data(mjm, mjd, nworld=args.nworld)
      rc = _make_context(mjm, args.nworld, args.width, args.height, use_shadows=case.use_shadows)
      enabled_mesh_geoms = sum(
        mjm.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and mjm.geom_group[i] in (0, 1, 2) for i in range(mjm.ngeom)
      )
      print(
        f"Case: {case.label}, use_shadows={case.use_shadows}, light_castshadow={case.light_castshadow}, "
        f"ngeom={mjm.ngeom}, nmesh={mjm.nmesh}, enabled_mesh_geoms={enabled_mesh_geoms}, nlight={mjm.nlight}"
      )
      timings.append(
        _benchmark(
          case.label,
          m,
          d,
          rc,
          args.nworld,
          args.width,
          args.height,
          args.warmup,
          args.iterations,
          include_step=not args.render_only and not args.skip_refit,
          include_refit=not args.skip_refit,
        )
      )

  for timing in timings:
    _print_timing(timing)

  by_label = {timing.label: timing for timing in timings}
  if "no_shadows" in by_label and "shadows" in by_label:
    print(f"shadow_slowdown={by_label['no_shadows'].world_fps / by_label['shadows'].world_fps:.2f}x")
  if "no_light_shadows" in by_label and "shadows" in by_label:
    print(f"shadow_ray_cost={by_label['no_light_shadows'].world_fps / by_label['shadows'].world_fps:.2f}x")


if __name__ == "__main__":
  main()
