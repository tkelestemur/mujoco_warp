"""Show exact BVH ray shadows and shadow-map shadows side-by-side.

This is a visualization helper for the shadow render benchmark scene. It opens
an OpenCV window and renders the same camera with:

* the existing exact BVH shadow-ray path
* the opt-in spot shadow-map path
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import warp as wp
from benchmark_shadow_render import SCENE
from benchmark_shadow_render import _find_fr3_dir
from benchmark_shadow_render import _load_model
from benchmark_shadow_render import _make_context
from benchmark_shadow_render import _make_fr3_scene

import mujoco_warp as mjw


def _render_context_rgb(
  m: mjw.Model,
  d: mjw.Data,
  rc: mjw.RenderContext,
  width: int,
  height: int,
  world: int,
) -> np.ndarray:
  mjw.render(m, d, rc)
  rgb_out = wp.zeros((d.nworld, height, width), dtype=wp.vec3)
  mjw.get_rgb(rc, 0, rgb_out)
  wp.synchronize()

  rgb = np.clip(rgb_out.numpy()[world], 0.0, 1.0)
  return (rgb * 255.0).astype(np.uint8)


def _label_image(rgb: np.ndarray, label: str) -> np.ndarray:
  bgr = rgb[:, :, ::-1].copy()
  bar_h = 34
  labeled = cv2.copyMakeBorder(bgr, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
  cv2.putText(
    labeled,
    label,
    (10, 23),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.62,
    (235, 235, 235),
    1,
    cv2.LINE_AA,
  )
  return labeled


def _build_scene(args: argparse.Namespace, tmpdir: Path) -> Path:
  if args.scene == "table":
    return SCENE

  fr3_dir = _find_fr3_dir(args.fr3_dir)
  return _make_fr3_scene(fr3_dir, tmpdir)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--scene", choices=("fr3", "table"), default="fr3")
  parser.add_argument("--fr3-dir", type=str, default=None, help="path to mujoco_menagerie/franka_fr3")
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument("--nworld", type=int, default=1)
  parser.add_argument("--world", type=int, default=0, help="world index to display")
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=240)
  parser.add_argument("--shadow-map-size", type=int, default=64)
  parser.add_argument("--shadow-map-bias", type=float, default=0.01)
  parser.add_argument("--use-textures", action="store_true")
  parser.add_argument("--step", action="store_true", help="advance physics before each rendered frame")
  parser.add_argument("--skip-refit", action="store_true", help="skip BVH refit between frames")
  return parser.parse_args()


def main():
  args = _parse_args()
  if args.world < 0 or args.world >= args.nworld:
    raise ValueError(f"--world must be in [0, {args.nworld})")

  global cv2
  try:
    import cv2
  except ImportError as exc:
    raise ImportError("OpenCV is required. Run with `uv run --with opencv-python ...`.") from exc

  wp.init()
  with tempfile.TemporaryDirectory(prefix="mjw_shadow_view_") as tmp, wp.ScopedDevice(args.device):
    scene = _build_scene(args, Path(tmp))
    mjm, mjd = _load_model(light_castshadow=True, scene=scene)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=args.nworld)

    exact_rc = _make_context(
      mjm,
      args.nworld,
      args.width,
      args.height,
      use_shadows=True,
      shadow_geom_groups=None,
      use_shadow_maps=False,
      shadow_map_size=args.shadow_map_size,
      shadow_map_bias=args.shadow_map_bias,
    )
    shadow_map_rc = _make_context(
      mjm,
      args.nworld,
      args.width,
      args.height,
      use_shadows=True,
      shadow_geom_groups=None,
      use_shadow_maps=True,
      shadow_map_size=args.shadow_map_size,
      shadow_map_bias=args.shadow_map_bias,
    )
    exact_rc.use_textures = args.use_textures
    shadow_map_rc.use_textures = args.use_textures

    if not args.skip_refit:
      mjw.refit_bvh(m, d, exact_rc)
      mjw.refit_bvh(m, d, shadow_map_rc)

    window = "mujoco_warp shadows: BVH rays vs shadow map"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
      if args.step:
        mjw.step(m, d)
      if not args.skip_refit:
        mjw.refit_bvh(m, d, exact_rc)
        mjw.refit_bvh(m, d, shadow_map_rc)

      exact_rgb = _render_context_rgb(m, d, exact_rc, args.width, args.height, args.world)
      shadow_map_rgb = _render_context_rgb(m, d, shadow_map_rc, args.width, args.height, args.world)

      left = _label_image(exact_rgb, "BVH shadow rays")
      right = _label_image(shadow_map_rgb, f"Shadow map {args.shadow_map_size}x{args.shadow_map_size}")
      cv2.imshow(window, np.concatenate([left, right], axis=1))

      key = cv2.waitKey(1 if args.step else 0) & 0xFF
      if key in (27, ord("q")):
        break

    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
