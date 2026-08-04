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

"""Warp kernels used by the OVRTX bridge."""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _usd_transform(position: wp.vec3, rotation: wp.mat33):
  """Convert a MuJoCo pose to a USD row-vector matrix."""
  return wp.mat44d(
    wp.float64(rotation[0, 0]),
    wp.float64(rotation[1, 0]),
    wp.float64(rotation[2, 0]),
    0.0,
    wp.float64(rotation[0, 1]),
    wp.float64(rotation[1, 1]),
    wp.float64(rotation[2, 1]),
    0.0,
    wp.float64(rotation[0, 2]),
    wp.float64(rotation[1, 2]),
    wp.float64(rotation[2, 2]),
    0.0,
    wp.float64(position[0]),
    wp.float64(position[1]),
    wp.float64(position[2]),
    1.0,
  )


@wp.kernel
def _pack_render_state(
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  cam_xpos_in: wp.array2d[wp.vec3],
  cam_xmat_in: wp.array2d[wp.mat33],
  # In:
  selected_geom_ids: wp.array[int],
  selected_camera_ids: wp.array[int],
  environment_offsets: wp.array[wp.vec3],
  geom_count: int,
  camera_count: int,
  # Out:
  positions_out: wp.array[wp.vec3],
  orientations_out: wp.array[wp.vec4],
  camera_transforms_out: wp.array[wp.mat44d],
):
  output_id = wp.tid()
  instance_count = geom_xpos_in.shape[0] * geom_count
  if output_id < instance_count:
    world_id = output_id // geom_count
    geom_id = selected_geom_ids[output_id - world_id * geom_count]
    positions_out[output_id] = geom_xpos_in[world_id, geom_id] + environment_offsets[world_id]
    orientation = wp.quat_from_matrix(geom_xmat_in[world_id, geom_id])
    orientations_out[output_id] = wp.vec4(
      orientation[0],
      orientation[1],
      orientation[2],
      orientation[3],
    )
  else:
    camera_transform_id = output_id - instance_count
    world_id = camera_transform_id // camera_count
    camera_id = selected_camera_ids[camera_transform_id - world_id * camera_count]
    camera_transforms_out[camera_transform_id] = _usd_transform(
      cam_xpos_in[world_id, camera_id],
      cam_xmat_in[world_id, camera_id],
    )


@wp.kernel
def _untile_rgba8(
  # In:
  atlas: wp.array3d[wp.uint8],
  tile_width: int,
  tile_height: int,
  columns: int,
  # Out:
  images_out: wp.array4d[wp.uint8],
):
  world_id, y, x = wp.tid()
  atlas_y = (world_id // columns) * tile_height + y
  atlas_x = (world_id % columns) * tile_width + x
  images_out[world_id, y, x, 0] = atlas[atlas_y, atlas_x, 0]
  images_out[world_id, y, x, 1] = atlas[atlas_y, atlas_x, 1]
  images_out[world_id, y, x, 2] = atlas[atlas_y, atlas_x, 2]
  images_out[world_id, y, x, 3] = atlas[atlas_y, atlas_x, 3]


def pack_render_state(
  geom_xpos: wp.array,
  geom_xmat: wp.array,
  camera_xpos: wp.array,
  camera_xmat: wp.array,
  geom_ids: wp.array,
  camera_ids: wp.array,
  environment_offsets: wp.array,
  positions: wp.array,
  orientations: wp.array,
  camera_transforms: wp.array,
  stream: wp.Stream | None = None,
) -> None:
  """Pack dynamic geom instances and cameras for one OVStage publication."""
  geom_count = geom_ids.shape[0]
  camera_count = camera_ids.shape[0]
  world_count = geom_xpos.shape[0]
  instance_count = world_count * geom_count
  camera_transform_count = world_count * camera_count
  if positions.shape[0] != instance_count or orientations.shape[0] != instance_count:
    raise ValueError(f"instance buffers have lengths {(positions.shape[0], orientations.shape[0])}, expected {instance_count}")
  if camera_transforms.shape[0] != camera_transform_count:
    raise ValueError(f"camera transform buffer has length {camera_transforms.shape[0]}, expected {camera_transform_count}")
  if environment_offsets.shape[0] != world_count:
    raise ValueError(f"environment offsets have length {environment_offsets.shape[0]}, expected {world_count}")
  output_count = instance_count + camera_transform_count
  if not output_count:
    return
  wp.launch(
    _pack_render_state,
    dim=output_count,
    inputs=[
      geom_xpos,
      geom_xmat,
      camera_xpos,
      camera_xmat,
      geom_ids,
      camera_ids,
      environment_offsets,
      geom_count,
      camera_count,
      positions,
      orientations,
      camera_transforms,
    ],
    device=geom_xpos.device,
    stream=stream,
  )


def untile_rgba8(
  atlas: wp.array,
  images: wp.array,
  tile_width: int,
  tile_height: int,
  columns: int,
  stream: wp.Stream | None = None,
) -> None:
  """Copy a tiled RGBA8 atlas to a contiguous ``[world, height, width, 4]`` tensor."""
  if atlas.dtype != wp.uint8 or atlas.ndim != 3 or atlas.shape[2] != 4:
    raise TypeError(f"expected an RGBA8 atlas, got dtype={atlas.dtype} shape={atlas.shape}")
  if images.dtype != wp.uint8 or images.ndim != 4 or images.shape[3] != 4:
    raise TypeError(f"expected an RGBA8 image batch, got dtype={images.dtype} shape={images.shape}")
  if images.shape[1:] != (tile_height, tile_width, 4):
    raise ValueError(f"image batch shape {images.shape} does not match tile size {(tile_width, tile_height)}")
  wp.launch(
    _untile_rgba8,
    dim=(images.shape[0], tile_height, tile_width),
    inputs=[atlas, tile_width, tile_height, columns, images],
    device=images.device,
    stream=stream,
  )
