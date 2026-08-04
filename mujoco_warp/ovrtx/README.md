# NVIDIA OVRTX renderer

This optional renderer couples MuJoCo Warp physics to
[NVIDIA OVRTX](https://github.com/NVIDIA-Omniverse/ovrtx) for photorealistic,
GPU-accelerated rendering across many simulation environments.

## Install

OVRTX currently requires Python 3.10-3.13, Linux x86-64, and a supported NVIDIA
GPU/driver. Install the optional dependencies in the MuJoCo Warp environment:

```bash
uv sync --extra ovrtx
```

The first process that creates an OVRTX renderer can take several minutes while
RTX shaders are compiled. Later runs reuse NVIDIA's shader cache.

## Use

```python
import mujoco

import mujoco_warp as mjw
from mujoco_warp.ovrtx import Renderer
from mujoco_warp.ovrtx import RendererConfig

mjm = mujoco.MjModel.from_xml_path("scene.xml")
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)

nworld = 256
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=nworld)
config = RendererConfig(width=128, height=128, camera_ids=(0,))

with Renderer(mjm, nworld, config) as renderer:
  frame = renderer.render(d)
  with frame.map() as output:
    atlas = output.atlas       # zero-copy CUDA atlas
    images = output.as_batch() # one GPU copy to [nworld, H, W, 4]
```

`atlas` is valid while its mapping is open and is ready on `output.stream`.
`images` owns its allocation and can outlive the mapping.

For pipelines that can run physics while RTX renders:

```python
pending = renderer.render_async(d)
mjw.step(m, d)
frame = pending.wait()
```

The transform upload snapshots the poses before `render_async` returns, so the
next physics step can safely update `d`.

Run the included renderer and throughput benchmark with:

```bash
uv run contrib/ovrtx_render.py benchmarks/render/primitives.xml \
  --nworld 256 --width 64 --height 64 --mode rtpt --overlap-physics
```

## Performance model

- Geometry and materials are exported once, kept as shared USD prototypes, and
  instance-referenced by each environment. Mesh buffers are not duplicated per
  world.
- Environments are separated spatially for efficient RTX acceleration-structure
  traversal. The default separation and per-camera far clip keep neighboring
  worlds out of primary camera rays; use `world_spacing` to tune unusually
  large or mobile-camera scenes.
- Only geoms affected by a joint or mocap ancestor are updated each frame.
  Static geoms stay in the cloned scene without consuming update bandwidth.
- All moving geoms across every world share one USD PointInstancer. One Warp
  kernel packs its `float32` positions and quaternions directly from MuJoCo
  Warp state, along with any moving-camera matrices.
- Two OVStage array writes publish every moving geom directly from CUDA.
  There is no per-geom Python loop or geometry device-to-host copy.
- Fixed cameras consume no per-frame update bandwidth. OVRTX 0.4 cannot combine
  CUDA-authored camera transforms with reliable PointInstancer streaming, so
  moving cameras use a reusable host staging buffer and OVStage's zero-copy map
  path. Prefer fixed cameras when maximizing very-large-batch throughput.
- One tiled RenderProduct per selected MuJoCo camera renders every environment.
  Keeping the tiled atlas avoids readback and untile overhead.
- A dedicated render stream uses CUDA events to order physics, transform ingest,
  output mapping, and optional untile work without device-wide synchronization.

The dynamic geometry payload is 28 bytes per moving geom/world: 12 bytes for
position and 16 bytes for orientation. OVStage's 4×4 double transform input is
used only for cameras that actually move, at 128 bytes per camera/world.

### NVIDIA L4 reference

With OVRTX 0.4 and OVStage 0.1 after kernel and shader warm-up:

| Workload | Result |
| --- | ---: |
| Publish 131,072 moving geoms across 8,192 worlds (3.67 MiB) | 0.169 ms median |
| Publish one moving camera in each of 8,192 worlds (1.00 MiB) | 4.13 ms median |
| Render 16 worlds × 125 moving geoms at 128×128 RTPT while overlapping physics | 45.5 ms/frame; 352 environment frames/s; 5.8 MP/s |

The first two rows isolate state publication from rendering. Treat these as
reference measurements rather than guarantees; scene complexity, resolution,
render settings, and driver caches dominate end-to-end throughput. Use the
included benchmark on the target workload to choose the render cadence.

## Rendering modes

- `RenderMode.REAL_TIME_PATH_TRACING` is the default photorealistic mode with
  real-time denoising.
- `RenderMode.PATH_TRACING` is progressive, reference-quality rendering.
- `RenderMode.MINIMAL` maximizes throughput when path-traced lighting is not
  required.

Additional OVRTX settings can be authored on every RenderProduct through
`RendererConfig.render_settings`.

## Current scope

Render-visible rigid MuJoCo geoms, meshes, height fields, textures, materials,
fixed cameras, and rigid-body motion are supported. Geoms hidden by MuJoCo's
visual geom-group mask (including collision-only groups) are intentionally
omitted. Scene topology and material assignments are shared and static after
construction. Dynamic flex/skin/tendon geometry and per-world material
randomization are not yet bridged to OVRTX.
