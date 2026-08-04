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

"""Configuration and layout helpers for the OVRTX renderer."""

from __future__ import annotations

import dataclasses
import enum
import math
import os
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any


class RenderMode(str, enum.Enum):
  """OVRTX camera render modes."""

  REAL_TIME_PATH_TRACING = "RealTimePathTracing"
  PATH_TRACING = "PathTracing"
  MINIMAL = "Minimal"


@dataclasses.dataclass(slots=True)
class RendererConfig:
  """Configuration for :class:`mujoco_warp.ovrtx.Renderer`.

  ``width`` and ``height`` are the resolution of one environment. OVRTX receives
  a single tiled render product per selected MuJoCo camera. ``world_spacing``
  controls separation inside the shared RTX scene; by default it is eight times
  MuJoCo's model extent.
  """

  width: int = 128
  height: int = 128
  camera_ids: Sequence[int] | None = None
  render_vars: Sequence[str] = ("LdrColor",)
  render_mode: RenderMode | str = RenderMode.REAL_TIME_PATH_TRACING
  device: str | None = None
  tile_columns: int | None = None
  world_spacing: float | None = None
  scene_directory: str | os.PathLike[str] | None = None
  dome_light_intensity: float = 500.0
  dome_light_texture: str | os.PathLike[str] | None = None
  distant_light_intensity: float = 3000.0
  distant_light_angle: float = 0.53
  distant_light_rotation: tuple[float, float, float] = (315.0, 0.0, 45.0)
  render_settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  keep_system_alive: bool = True
  log_level: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class TileLayout:
  """Layout of the multi-environment output atlas."""

  world_count: int
  columns: int
  rows: int
  tile_width: int
  tile_height: int

  @property
  def atlas_width(self) -> int:
    return self.columns * self.tile_width

  @property
  def atlas_height(self) -> int:
    return self.rows * self.tile_height


def choose_tile_layout(
  world_count: int,
  tile_width: int,
  tile_height: int,
  columns: int | None = None,
) -> TileLayout:
  """Choose a compact row-major atlas layout."""
  if world_count <= 0:
    raise ValueError(f"world_count must be positive, got {world_count}")
  if tile_width <= 0 or tile_height <= 0:
    raise ValueError(f"tile dimensions must be positive, got {(tile_width, tile_height)}")
  if columns is None:
    columns = math.ceil(math.sqrt(world_count))
  if columns <= 0:
    raise ValueError(f"tile_columns must be positive, got {columns}")
  columns = min(columns, world_count)
  rows = math.ceil(world_count / columns)
  return TileLayout(
    world_count=world_count,
    columns=columns,
    rows=rows,
    tile_width=tile_width,
    tile_height=tile_height,
  )


def normalize_camera_ids(camera_ids: Sequence[int] | None, camera_count: int) -> tuple[int, ...]:
  """Validate and normalize selected MuJoCo camera IDs."""
  if camera_count <= 0:
    raise ValueError("the MuJoCo model must contain at least one camera")
  result = tuple(range(camera_count)) if camera_ids is None else tuple(int(camera_id) for camera_id in camera_ids)
  if not result:
    raise ValueError("camera_ids must select at least one camera")
  if len(set(result)) != len(result):
    raise ValueError(f"camera_ids contains duplicates: {result}")
  invalid = [camera_id for camera_id in result if camera_id < 0 or camera_id >= camera_count]
  if invalid:
    raise ValueError(f"camera IDs {invalid} are outside [0, {camera_count})")
  return result


def normalize_render_vars(render_vars: Sequence[str]) -> tuple[str, ...]:
  """Validate requested OVRTX render variables."""
  result = tuple(str(render_var) for render_var in render_vars)
  if not result:
    raise ValueError("render_vars must not be empty")
  if len(set(result)) != len(result):
    raise ValueError(f"render_vars contains duplicates: {result}")
  if any(not render_var for render_var in result):
    raise ValueError("render_vars must not contain empty names")
  return result


def normalize_render_mode(render_mode: RenderMode | str) -> str:
  """Return an OVRTX render-mode token."""
  value = render_mode.value if isinstance(render_mode, RenderMode) else str(render_mode)
  supported = {mode.value for mode in RenderMode}
  if value not in supported:
    raise ValueError(f"unsupported render mode {value!r}; expected one of {sorted(supported)}")
  return value
