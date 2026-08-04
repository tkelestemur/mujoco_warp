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

"""GPU-native bridge between MuJoCo Warp state and NVIDIA OVRTX."""

from __future__ import annotations

import ctypes
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import warp as wp

from mujoco_warp.ovrtx._config import RendererConfig
from mujoco_warp.ovrtx._config import choose_tile_layout
from mujoco_warp.ovrtx._config import normalize_camera_ids
from mujoco_warp.ovrtx._config import normalize_render_mode
from mujoco_warp.ovrtx._config import normalize_render_vars
from mujoco_warp.ovrtx._kernels import pack_render_state
from mujoco_warp.ovrtx._kernels import untile_rgba8
from mujoco_warp.ovrtx._scene import build_scene
from mujoco_warp.ovrtx._scene import populate_scene


def _load_runtime():
  try:
    import ovrtx
    import ovstage
  except ImportError as error:
    raise ImportError(
      "NVIDIA OVRTX and OVStage are required. With Python 3.10-3.13 on a Linux x86_64 host "
      "with an NVIDIA GPU, install the optional dependencies with `uv sync --extra ovrtx`."
    ) from error
  return ovrtx, ovstage


def _packed_dltensor(ovstage: Any, array: wp.array, lanes: int):
  """Describe a Warp composite array with OVStage's packed lane type."""
  tensor = ovstage.DLTensor.from_dlpack(array)
  component_count = int(np.prod(tensor.shape_tuple[1:]))
  if component_count != lanes:
    raise ValueError(f"expected {lanes} DLPack components per row, got shape {tensor.shape_tuple}")
  tensor.dtype = ovstage.DLDataType(code=tensor.dtype.code, bits=tensor.dtype.bits, lanes=lanes)
  tensor.ndim = 1
  tensor._shape_storage = (ctypes.c_int64 * 1)(array.shape[0])
  tensor.shape = ctypes.cast(tensor._shape_storage, ctypes.POINTER(ctypes.c_int64))
  tensor.strides = None
  return tensor


class MappedOutput:
  """A CUDA mapping of one tiled OVRTX render variable.

  The zero-copy :attr:`atlas` is valid inside this object's lifetime. Use
  :meth:`as_batch` when a persistent ``[world, height, width, 4]`` RGBA8 tensor
  is more convenient.
  """

  def __init__(
    self,
    owner: Renderer,
    render_var_output: Any,
    render_var: str,
    consumer_stream: wp.Stream | None,
  ):
    self._owner = owner
    self._render_var = render_var
    self._stream = consumer_stream or owner.stream
    if self._stream.device != owner.device:
      raise ValueError(f"consumer stream is on {self._stream.device}, but the renderer is on {owner.device}")
    self._mapping = render_var_output.map(
      device=owner._ovrtx.Device.CUDA,
      sync_stream=self._stream.cuda_stream,
    )
    self._atlas = wp.from_dlpack(self._mapping)
    self._batch = None
    self._release_event = None
    self._closed = False
    owner._mappings.add(self)

  @property
  def stream(self) -> wp.Stream:
    """Warp stream on which the atlas is ready and batch copies are launched."""
    return self._stream

  @property
  def atlas(self) -> wp.array:
    """Zero-copy OVRTX output atlas."""
    if self._closed:
      raise RuntimeError("the OVRTX output mapping is closed")
    return self._atlas

  def as_batch(self, output: wp.array | None = None) -> wp.array:
    """Untile an RGBA8 atlas into a contiguous per-world image batch.

    The returned array owns its memory and remains valid after this mapping is
    closed. The copy stays entirely on the GPU.
    """
    if self._closed:
      raise RuntimeError("the OVRTX output mapping is closed")
    layout = self._owner.layout.tile_layout
    expected_shape = (layout.atlas_height, layout.atlas_width, 4)
    if self._atlas.shape != expected_shape:
      raise RuntimeError(
        f"OVRTX returned atlas shape {self._atlas.shape}, expected {expected_shape}; "
        "the installed OVRTX tiled-layout policy is incompatible with this bridge"
      )
    if output is None:
      if self._batch is None:
        self._batch = wp.empty(
          (layout.world_count, layout.tile_height, layout.tile_width, 4),
          dtype=wp.uint8,
          device=self._owner.device,
        )
      output = self._batch
    if output.device != self._owner.device:
      raise ValueError(f"output is on {output.device}, expected {self._owner.device}")
    expected_output_shape = (layout.world_count, layout.tile_height, layout.tile_width, 4)
    if output.dtype != wp.uint8 or output.shape != expected_output_shape:
      raise ValueError(f"output must be an RGBA8 array with shape {expected_output_shape}, got {output.dtype} {output.shape}")
    untile_rgba8(
      self._atlas,
      output,
      layout.tile_width,
      layout.tile_height,
      layout.columns,
      stream=self._stream,
    )
    return output

  def close(self) -> None:
    """Release the OVRTX mapping after queued CUDA consumers finish."""
    if self._closed:
      return
    self._closed = True
    mapping = self._mapping
    try:
      release_event = self._stream.record_event()
      self._release_event = release_event
      # A DLPack-derived Warp view can extend the mapping lifetime. Pinning the
      # Event on the native mapping keeps its raw CUDA handle valid until the
      # deferred unmap actually runs.
      mapping._mujoco_warp_release_event = release_event
      mapping.unmap(event=release_event.cuda_event)
    finally:
      self._atlas = None
      self._mapping = None
      owner = self._owner
      self._owner = None
      if owner is not None:
        owner._mappings.discard(self)

  def __enter__(self) -> MappedOutput:
    return self

  def __exit__(self, *_exc) -> None:
    self.close()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass


class RenderFrame:
  """Completed outputs from one OVRTX render step."""

  def __init__(self, owner: Renderer, outputs: Any, ordinal: int):
    self._owner = owner
    self._outputs = outputs
    self.ordinal = ordinal

  @property
  def camera_ids(self) -> tuple[int, ...]:
    return self._owner.layout.camera_ids

  def map(
    self,
    render_var: str = "LdrColor",
    *,
    camera_id: int | None = None,
    consumer_stream: wp.Stream | None = None,
  ) -> MappedOutput:
    """Map one camera's tiled output to CUDA without a host copy."""
    if self._outputs is None:
      raise RuntimeError("this render frame is closed")
    if camera_id is None:
      if len(self.camera_ids) != 1:
        raise ValueError(f"camera_id is required when rendering cameras {self.camera_ids}")
      camera_id = self.camera_ids[0]
    if camera_id not in self._owner.layout.product_paths:
      raise ValueError(f"camera {camera_id} was not selected; available cameras are {self.camera_ids}")
    if render_var not in self._owner.layout.render_vars:
      raise ValueError(f"render var {render_var!r} was not requested; available vars are {self._owner.layout.render_vars}")

    product = self._outputs[self._owner.layout.product_paths[camera_id]]
    if not product.frames:
      raise RuntimeError(f"OVRTX produced no frame for camera {camera_id}")
    frame = product.frames[-1]
    try:
      output = frame.render_vars[render_var]
    except KeyError:
      available = tuple(frame.render_vars)
      raise RuntimeError(f"OVRTX did not return {render_var!r}; available vars are {available}") from None
    return MappedOutput(self._owner, output, render_var, consumer_stream)

  def close(self) -> None:
    """Release the native step-result container."""
    self._outputs = None

  def __enter__(self) -> RenderFrame:
    return self

  def __exit__(self, *_exc) -> None:
    self.close()


class PendingFrame:
  """An OVRTX render step that can overlap subsequent physics work."""

  def __init__(self, owner: Renderer, operation: Any, ordinal: int):
    self._owner = owner
    self._operation = operation
    self._pending_fetch = None
    self._frame = None
    self.ordinal = ordinal

  def wait(self, timeout_ns: int | None = None) -> RenderFrame | None:
    """Wait for rendering and fetch its outputs, or return ``None`` on timeout."""
    if self._frame is not None:
      return self._frame
    try:
      if self._pending_fetch is None:
        self._pending_fetch = self._operation.wait(timeout_ns=timeout_ns)
        if self._pending_fetch is None:
          return None
      outputs = self._pending_fetch.fetch(timeout_ns=timeout_ns)
      if outputs is None:
        return None
      self._frame = RenderFrame(self._owner, outputs, self.ordinal)
      self._operation = None
      self._pending_fetch = None
      self._owner._finish_pending(self)
      return self._frame
    except Exception:
      self._owner._finish_pending(self)
      raise

  def query_status(self):
    """Return an OVRTX progress snapshot while the step is pending."""
    if self._operation is None:
      raise RuntimeError("the render step has completed")
    return self._operation.query_status()


class Renderer:
  """Photorealistic batched renderer backed by NVIDIA OVRTX.

  Static scene content is cloned once per MuJoCo Warp world. Moving rigid geoms
  use compact PointInstancer arrays, while dynamic cameras use matrix updates.
  One Warp kernel prepares both directly from device-resident simulation state.
  """

  def __init__(
    self,
    model: mujoco.MjModel,
    world_count: int,
    config: RendererConfig | None = None,
    *,
    stream: wp.Stream | None = None,
  ):
    if world_count <= 0:
      raise ValueError(f"world_count must be positive, got {world_count}")
    self.model = model
    self.world_count = world_count
    self.config = config or RendererConfig()
    self._ovrtx, self._ovstage = _load_runtime()
    self.device = wp.get_device(self.config.device)
    if not self.device.is_cuda:
      raise RuntimeError(f"OVRTX requires a CUDA device, got {self.device}")
    self.stream = stream or wp.Stream(self.device)
    if self.stream.device != self.device:
      raise ValueError(f"render stream is on {self.stream.device}, expected {self.device}")

    camera_ids = normalize_camera_ids(self.config.camera_ids, model.ncam)
    render_vars = normalize_render_vars(self.config.render_vars)
    render_mode = normalize_render_mode(self.config.render_mode)
    tile_layout = choose_tile_layout(
      world_count,
      self.config.width,
      self.config.height,
      self.config.tile_columns,
    )

    self._temporary_directory = None
    if self.config.scene_directory is None:
      self._temporary_directory = tempfile.TemporaryDirectory(prefix="mujoco-warp-ovrtx-")
      scene_root = Path(self._temporary_directory.name)
    else:
      scene_root = Path(self.config.scene_directory).expanduser().resolve()

    self._native_renderer = None
    self._stage = None
    self._paths = None
    self._path_lists = []
    self._queries = []
    self._instance_query = None
    self._camera_query = None
    self._camera_path_list = None
    self._camera_path_ids = ()
    self._camera_path_indices = {}
    self._camera_map_offsets = {}
    self._positions_attribute = None
    self._orientations_attribute = None
    self._transform_attribute = None
    self._geom_ids = None
    self._camera_ids = None
    self._environment_offsets = None
    self._instance_positions = None
    self._instance_orientations = None
    self._camera_transforms = None
    self._camera_transforms_host = None
    self._camera_transforms_host_view = None
    self._instance_writes = ()
    self._source_event = None
    self._pack_event = None
    self._camera_copy_event = None
    self._mappings = set()
    self._pending = None
    self._closed = False
    try:
      self.layout = build_scene(
        model,
        world_count,
        camera_ids,
        render_vars,
        render_mode,
        tile_layout,
        scene_root,
        self.device.ordinal,
        world_spacing=self.config.world_spacing,
        dome_light_intensity=self.config.dome_light_intensity,
        dome_light_texture=self.config.dome_light_texture,
        distant_light_intensity=self.config.distant_light_intensity,
        distant_light_angle=self.config.distant_light_angle,
        distant_light_rotation=self.config.distant_light_rotation,
        render_settings=self.config.render_settings,
      )
      native_config = self._ovrtx.RendererConfig(
        # OVRTX 0.4 does not reliably stream dynamic PointInstancer arrays
        # while GPU world-transform propagation is enabled.
        read_gpu_transforms=False,
        keep_system_alive=self.config.keep_system_alive,
        active_cuda_gpus=str(self.device.ordinal),
        log_level=self.config.log_level,
      )
      self._native_renderer = self._ovrtx.Renderer(config=native_config)
      self._stage = self._ovstage.Stage("mujoco_warp.ovrtx")
      self._native_renderer.attach_ovstage(self._stage)
      self._ordinal = populate_scene(self._stage, self.layout, self._ovstage)

      has_updates = self.layout.point_instancer_path is not None or bool(self.layout.camera_transform_paths)
      if has_updates:
        self._paths = self._ovstage.PathDictionary(self._stage)
        if self.layout.point_instancer_path is not None:
          path_list = self._paths.create_path_list_from_strings([self.layout.point_instancer_path])
          self._path_lists.append(path_list)
          self._instance_query = self._stage.query_from_path_list(path_list)
          self._queries.append(self._instance_query)
          self._positions_attribute = self._paths.intern_token("positions")
          self._orientations_attribute = self._paths.intern_token("orientationsf")

        if self.layout.camera_transform_paths:
          path_list = self._paths.create_path_list_from_strings(self.layout.camera_transform_paths)
          self._path_lists.append(path_list)
          self._camera_path_list = path_list
          self._camera_path_ids = tuple(self._paths.get_paths(path_list))
          self._camera_path_indices = {path_id: index for index, path_id in enumerate(self._camera_path_ids)}
          self._camera_query = self._stage.query_from_path_list(path_list)
          self._queries.append(self._camera_query)
          self._transform_attribute = self._paths.intern_token("omni:xform")

        self._geom_ids = wp.array(
          np.asarray(self.layout.dynamic_geom_ids, dtype=np.int32),
          dtype=wp.int32,
          device=self.device,
        )
        self._camera_ids = wp.array(
          np.asarray(self.layout.update_camera_ids, dtype=np.int32),
          dtype=wp.int32,
          device=self.device,
        )
        self._environment_offsets = wp.array(
          np.asarray(self.layout.environment_offsets, dtype=np.float32),
          dtype=wp.vec3,
          device=self.device,
        )
        self._instance_positions = wp.empty(self.layout.instance_count, dtype=wp.vec3, device=self.device)
        self._instance_orientations = wp.empty(self.layout.instance_count, dtype=wp.vec4, device=self.device)
        camera_transform_count = world_count * len(self.layout.update_camera_ids)
        self._camera_transforms = wp.empty(camera_transform_count, dtype=wp.mat44d, device=self.device)
        self._source_event = wp.Event(self.device)
        self._pack_event = wp.Event(self.device)

        if self._instance_query is not None:
          position_tensor = _packed_dltensor(self._ovstage, self._instance_positions, 3)
          orientation_tensor = _packed_dltensor(self._ovstage, self._instance_orientations, 4)
          self._instance_writes = (
            self._ovstage.WriteDesc(
              attribute=self._positions_attribute,
              tensors=position_tensor,
              is_array=True,
              semantic=self._ovstage.AttributeSemantic.POINT,
              cuda_event=self._pack_event.cuda_event,
              cuda_stream=self.stream.cuda_stream,
            ),
            self._ovstage.WriteDesc(
              attribute=self._orientations_attribute,
              tensors=orientation_tensor,
              is_array=True,
              semantic=self._ovstage.AttributeSemantic.QUATERNION,
              cuda_event=self._pack_event.cuda_event,
              cuda_stream=self.stream.cuda_stream,
            ),
          )
        if self._camera_query is not None:
          # OVRTX 0.4 only propagates CUDA-authored omni:xform values when
          # read_gpu_transforms is enabled, but that mode drops streamed
          # PointInstancer updates. Cameras are normally a tiny payload, so
          # stage them through a reusable host buffer and OVStage's zero-copy
          # map path. Copying CUDA directly into a fresh mapped CPU allocation
          # is substantially slower because the allocation changes each frame.
          self._camera_transforms_host = wp.empty(camera_transform_count, dtype=wp.mat44d, device="cpu")
          self._camera_transforms_host_view = np.from_dlpack(self._camera_transforms_host).reshape(
            camera_transform_count,
            16,
          )
          self._camera_copy_event = wp.Event(self.device)
    except Exception:
      self.close()
      raise

  @property
  def scene_path(self) -> Path:
    """Generated USD scene path."""
    return self.layout.scene_path

  def _validate_data(self, data: Any) -> None:
    expected_shapes = {
      "geom_xpos": (self.world_count, self.model.ngeom),
      "geom_xmat": (self.world_count, self.model.ngeom),
      "cam_xpos": (self.world_count, self.model.ncam),
      "cam_xmat": (self.world_count, self.model.ncam),
    }
    for name, expected_shape in expected_shapes.items():
      array = getattr(data, name)
      if array.shape != expected_shape:
        raise ValueError(f"data.{name} has shape {array.shape}, expected {expected_shape}")
      if array.device != self.device:
        raise ValueError(f"data.{name} is on {array.device}, renderer is on {self.device}")

  def _publish_camera_transforms(self, ordinal: int) -> None:
    mapping = None
    ready = False
    copy_synchronized = False
    try:
      mapping = self._stage.map_attribute(
        self._camera_query,
        self._transform_attribute,
        ordinal=ordinal,
      )
      mapping.wait()
      ready = True
      groups = []
      while True:
        group = mapping.fetch_next()
        if group is None:
          break
        groups.append(group)

      wp.synchronize_event(self._camera_copy_event)
      copy_synchronized = True
      ranges = []
      for group_index, group in enumerate(groups):
        if group.tensor_count != 1 or group.has_data_index_map or group.data_count != group.prim_count:
          raise RuntimeError("OVStage returned an incompatible dynamic-camera data layout")
        signature = (
          group_index,
          group.prim_offset,
          group.prim_count,
          group.has_prim_index_map,
          group.meta.layout_generation,
        )
        start = self._camera_map_offsets.get(signature)
        if start is None:
          group_path_ids = self._paths.get_paths(group.prim_list)
          source_indices = [
            self._camera_path_indices[group_path_ids[group.prim_index(local)]] for local in range(group.prim_count)
          ]
          start = source_indices[0] if source_indices else 0
          if source_indices != list(range(start, start + group.prim_count)):
            raise RuntimeError("OVStage returned a non-contiguous dynamic-camera map")
          self._camera_map_offsets[signature] = start
        end = start + group.prim_count
        target = np.from_dlpack(group.dlpack(0))
        expected_shape = (group.prim_count, 16)
        if target.shape != expected_shape or target.dtype != np.float64:
          raise RuntimeError(
            f"OVStage returned camera matrix buffer {target.shape} {target.dtype}, expected {expected_shape} float64"
          )
        np.copyto(target, self._camera_transforms_host_view[start:end])
        ranges.append((start, end))

      cursor = 0
      for start, end in sorted(ranges):
        if start != cursor:
          raise RuntimeError("OVStage dynamic-camera map did not cover the camera query exactly once")
        cursor = end
      if cursor != self._camera_transforms.shape[0]:
        raise RuntimeError("OVStage dynamic-camera map did not cover the camera query exactly once")

      mapping.unmap().wait()
      ready = False
    finally:
      if not copy_synchronized:
        wp.synchronize_event(self._camera_copy_event)
      if ready:
        try:
          mapping.unmap().wait()
        except Exception:
          pass

  def _publish_state(self, data: Any, source_stream: wp.Stream | None) -> int:
    self._validate_data(data)
    if not self._queries:
      return self._ordinal
    source_stream = source_stream or wp.get_stream(self.device)
    if source_stream.device != self.device:
      raise ValueError(f"source stream is on {source_stream.device}, expected {self.device}")
    if source_stream.cuda_stream != self.stream.cuda_stream:
      self.stream.wait_stream(source_stream, event=self._source_event)

    pack_render_state(
      data.geom_xpos,
      data.geom_xmat,
      data.cam_xpos,
      data.cam_xmat,
      self._geom_ids,
      self._camera_ids,
      self._environment_offsets,
      self._instance_positions,
      self._instance_orientations,
      self._camera_transforms,
      stream=self.stream,
    )
    self.stream.record_event(self._pack_event)
    ordinal = self._ordinal + 1
    if self._camera_query is not None:
      wp.copy(self._camera_transforms_host, self._camera_transforms, stream=self.stream)
      self.stream.record_event(self._camera_copy_event)
    if self._instance_query is not None:
      self._stage.write_attributes(self._instance_query, self._instance_writes, ordinal=ordinal).wait()
    if self._camera_query is not None:
      self._publish_camera_transforms(ordinal)
    # Attached OVRTX steps are gated by OVStage's global floor. Attribute-
    # scoped floors do not satisfy step_with_stage's ordinal precondition.
    self._stage.advance_write_floor(ordinal, self._ovstage.Scope.ALL).wait()
    self._ordinal = ordinal
    return ordinal

  def _enqueue_step(self, ordinal: int, delta_time: float) -> PendingFrame:
    operation = self._native_renderer.step_async(
      render_products=set(self.layout.product_paths.values()),
      delta_time=delta_time,
      ordinal=ordinal,
    )
    pending = PendingFrame(self, operation, ordinal)
    self._pending = pending
    return pending

  def _normalize_delta_time(self, delta_time: float | None) -> float:
    if delta_time is None:
      delta_time = float(self.model.opt.timestep)
    if delta_time < 0.0:
      raise ValueError(f"delta_time must be non-negative, got {delta_time}")
    return delta_time

  def render_async(
    self,
    data: Any,
    *,
    delta_time: float | None = None,
    source_stream: wp.Stream | None = None,
  ) -> PendingFrame:
    """Publish state and enqueue OVRTX rendering without waiting for the image."""
    if self._closed:
      raise RuntimeError("the OVRTX renderer is closed")
    if self._pending is not None:
      raise RuntimeError("a render step is already pending; call pending.wait() before enqueueing another")
    delta_time = self._normalize_delta_time(delta_time)
    ordinal = self._publish_state(data, source_stream)
    return self._enqueue_step(ordinal, delta_time)

  def render_current_async(self, *, delta_time: float | None = None) -> PendingFrame:
    """Render the last published state again without another transform upload."""
    if self._closed:
      raise RuntimeError("the OVRTX renderer is closed")
    if self._pending is not None:
      raise RuntimeError("a render step is already pending; call pending.wait() before enqueueing another")
    return self._enqueue_step(self._ordinal, self._normalize_delta_time(delta_time))

  def render(
    self,
    data: Any,
    *,
    delta_time: float | None = None,
    source_stream: wp.Stream | None = None,
  ) -> RenderFrame:
    """Render synchronously and return the completed frame."""
    frame = self.render_async(data, delta_time=delta_time, source_stream=source_stream).wait()
    if frame is None:
      raise RuntimeError("an infinite OVRTX render wait unexpectedly timed out")
    return frame

  def render_current(self, *, delta_time: float | None = None) -> RenderFrame:
    """Render the last published state again without another transform upload."""
    frame = self.render_current_async(delta_time=delta_time).wait()
    if frame is None:
      raise RuntimeError("an infinite OVRTX render wait unexpectedly timed out")
    return frame

  def warmup(
    self,
    data: Any,
    frames: int = 4,
    *,
    delta_time: float | None = None,
    source_stream: wp.Stream | None = None,
  ) -> None:
    """Warm shader caches, texture streaming, and real-time denoising history."""
    if frames < 0:
      raise ValueError(f"frames must be non-negative, got {frames}")
    if not frames:
      return
    frame = self.render(data, delta_time=delta_time, source_stream=source_stream)
    frame.close()
    for _ in range(frames - 1):
      frame = self.render_current(delta_time=delta_time)
      frame.close()

  def _finish_pending(self, pending: PendingFrame) -> None:
    if self._pending is pending:
      self._pending = None

  def close(self) -> None:
    """Release OVRTX, OVStage, CUDA, and temporary scene resources."""
    if self._closed:
      return
    self._closed = True
    errors = []

    if self._pending is not None:
      try:
        frame = self._pending.wait()
        if frame is not None:
          frame.close()
      except Exception as error:
        errors.append(error)
      self._pending = None

    for mapping in tuple(self._mappings):
      try:
        mapping.close()
      except Exception as error:
        errors.append(error)
    self._mappings.clear()

    for query in reversed(self._queries):
      try:
        query.release().wait()
      except Exception as error:
        errors.append(error)
    self._queries.clear()
    self._instance_query = None
    self._camera_query = None
    if self._paths is not None:
      for path_list in reversed(self._path_lists):
        try:
          self._paths.destroy_path_list(path_list)
        except Exception as error:
          errors.append(error)
    self._path_lists.clear()
    if self._paths is not None:
      try:
        self._paths.destroy()
      except Exception as error:
        errors.append(error)
      self._paths = None
    if self._native_renderer is not None:
      try:
        self._native_renderer.destroy()
      except Exception as error:
        errors.append(error)
      self._native_renderer = None
    if self._stage is not None:
      try:
        self._stage.destroy()
      except Exception as error:
        errors.append(error)
      self._stage = None
    if self._temporary_directory is not None:
      try:
        self._temporary_directory.cleanup()
      except Exception as error:
        errors.append(error)
      self._temporary_directory = None

    if errors:
      details = "; ".join(str(error) for error in errors)
      raise RuntimeError(f"errors while closing the OVRTX renderer: {details}")

  def __enter__(self) -> Renderer:
    return self

  def __exit__(self, *_exc) -> None:
    self.close()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
