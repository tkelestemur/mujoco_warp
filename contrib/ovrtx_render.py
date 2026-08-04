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

"""Render and benchmark a MuJoCo Warp model with NVIDIA OVRTX."""

import argparse
import time
from pathlib import Path

import mujoco
import warp as wp
from PIL import Image

import mujoco_warp as mjw
from mujoco_warp.ovrtx import Renderer
from mujoco_warp.ovrtx import RendererConfig
from mujoco_warp.ovrtx import RenderMode


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("model", type=Path, help="MJCF or MJB model path")
  parser.add_argument("--nworld", type=int, default=16, help="number of parallel environments")
  parser.add_argument("--camera", type=int, default=0, help="MuJoCo camera ID")
  parser.add_argument("--width", type=int, default=128, help="per-environment image width")
  parser.add_argument("--height", type=int, default=128, help="per-environment image height")
  parser.add_argument("--frames", type=int, default=20, help="number of timed frames")
  parser.add_argument("--warmup", type=int, default=4, help="untimed warm-up frames")
  parser.add_argument(
    "--mode",
    choices=("rtpt", "pathtracing", "minimal"),
    default="rtpt",
    help="quality/performance mode",
  )
  parser.add_argument("--hdri", type=Path, help="optional latitude-longitude HDR environment map")
  parser.add_argument("--output", type=Path, default=Path("ovrtx.png"), help="output PNG path")
  parser.add_argument("--world", type=int, default=0, help="environment ID to save")
  parser.add_argument(
    "--world-spacing",
    type=float,
    help="distance between environments in the shared RTX scene",
  )
  parser.add_argument(
    "--overlap-physics",
    action="store_true",
    help="enqueue the next physics step while OVRTX renders",
  )
  return parser.parse_args()


def _load_model(path: Path) -> mujoco.MjModel:
  if path.suffix == ".mjb":
    return mujoco.MjModel.from_binary_path(str(path))
  return mujoco.MjModel.from_xml_path(str(path))


def main() -> None:
  args = _parse_args()
  if args.frames <= 0 or args.warmup < 0:
    raise ValueError("--frames must be positive and --warmup must be non-negative")
  if args.world < 0 or args.world >= args.nworld:
    raise ValueError(f"--world must be in [0, {args.nworld})")

  modes = {
    "rtpt": RenderMode.REAL_TIME_PATH_TRACING,
    "pathtracing": RenderMode.PATH_TRACING,
    "minimal": RenderMode.MINIMAL,
  }
  model = _load_model(args.model)
  cpu_data = mujoco.MjData(model)
  mujoco.mj_forward(model, cpu_data)
  warp_model = mjw.put_model(model)
  data = mjw.put_data(model, cpu_data, nworld=args.nworld)
  config = RendererConfig(
    width=args.width,
    height=args.height,
    camera_ids=(args.camera,),
    render_mode=modes[args.mode],
    dome_light_texture=args.hdri,
    world_spacing=args.world_spacing,
  )

  if args.overlap_physics:
    print("Compiling and warming one MuJoCo Warp physics step.")
    mjw.step(warp_model, data)
    wp.synchronize_device(data.qpos.device)

  print("Creating OVRTX renderer. A first-ever run may spend a few minutes compiling RTX shaders.")
  with Renderer(model, args.nworld, config) as renderer:
    renderer.warmup(data, frames=args.warmup)

    start = time.perf_counter()
    for _ in range(args.frames):
      if args.overlap_physics:
        pending = renderer.render_async(data)
        mjw.step(warp_model, data)
        frame = pending.wait()
      else:
        frame = renderer.render(data)
      frame.close()
    elapsed = time.perf_counter() - start

    with renderer.render(data) as frame:
      with frame.map() as output:
        images = output.as_batch()
    pixels = images.numpy()

  args.output.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(pixels[args.world]).save(args.output)
  environment_fps = args.frames * args.nworld / elapsed
  megapixels_per_second = environment_fps * args.width * args.height / 1.0e6
  label = "render + overlapped physics" if args.overlap_physics else "render"
  print(f"{label}: {elapsed / args.frames * 1000.0:.3f} ms/frame")
  print(f"throughput: {environment_fps:,.0f} environment frames/s ({megapixels_per_second:,.1f} MP/s)")
  print(f"saved environment {args.world} to {args.output}")


if __name__ == "__main__":
  main()
