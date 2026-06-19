# Offline Table Textures

This directory contains file-backed RGB PNG textures for MuJoCo Warp table
texture randomization demos. The textures are saved offline and are loaded by
MuJoCo through `<texture file="...">`; they are not generated procedurally at
runtime.

All images were converted to 512x512 RGB PNGs so MuJoCo, Warp, and Viser use a
consistent format.

## Sources

- `robosuite_*.png`: derived from `ARISE-Initiative/robosuite` texture assets,
  MIT License.
- `polyhaven_*.png`: derived from Poly Haven diffuse maps, CC0-1.0.

The IsaacSim-style GR00T example that motivated this dataset lists remote NVIDIA
MDL material URLs for wood, plastic, paint, and metal materials. Those MDL
assets are useful source references, but they are not vendored here because the
MuJoCo Warp demo needs plain image files and the NVIDIA Omniverse material
license is separate from the GR00T repository's Apache-2.0 code license.

## MIT License Notice for robosuite Assets

MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and
Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
