# 2026-06-25 01 SIBR GLEW WSLg Compatibility

## Context

The user ran `code/scripts/run_sibr_gaussian_viewer.sh` locally and reached OpenGL context creation, but SIBR aborted at GLEW initialization:

```text
OpenGL Version: 4.5 (Compatibility Profile) Mesa 25.2.8-0ubuntu0.24.04.2
cannot initialize GLEW (used to load OpenGL function)
```

This occurred after several EGL/Mesa warnings from WSLg.

## Diagnosis

SIBR was built with `GLEW_EGL` because EGL was found during CMake configure. In this mode, GLEW can report `GLEW_ERROR_NO_GLX_DISPLAY` even after a valid OpenGL context has been created. SIBR already had a commented note about this behavior for offscreen contexts, but the local viewer is run interactively.

## Change

Patched `external/graphdeco-gaussian-splatting/SIBR_viewers/src/core/graphics/Window.cpp` so that:

- `GLEW_ERROR_NO_GLX_DISPLAY` is treated as a warning under `GLEW_EGL`.
- Other GLEW errors still abort.
- The fatal GLEW path now prints `glewGetErrorString(err)` for better diagnostics.

Rebuilt and reinstalled SIBR:

```bash
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

## Verification

- Rebuild/install succeeded.
- `libsibr_graphics.so` contains the new warning string:

```text
GLEW reported no GLX display after context creation; continuing for EGL/WSLg compatibility.
```

Codex still cannot fully launch the GUI because its execution sandbox fails earlier at GLFW platform detection, but this is distinct from the user's local GLEW failure.

## Next

Ask the user to rerun:

```bash
code/scripts/run_sibr_gaussian_viewer.sh
```

## Follow-Up: Duplicate-With-Keys Failure Narrowed

The user reran the viewer after the previous stage-specific CUDA checks. The render now fails at a narrower point:

```text
CudaRasterizer forward failed after forward/duplicate_with_keys: an illegal memory access was encountered
```

This means the viewer has already passed scene loading, SfM mesh loading, Gaussian loading, preprocessing, tile-touch prefix sum, and reading `num_rendered`. The crash is now localized to the Gaussian-to-tile duplication stage, where per-Gaussian tile ranges are expanded into sortable `(tile, depth)` keys.

Additional diagnostic/safety updates:

- Added `CudaRasterizer forward stats` logging for `P`, render resolution, tile grid size, and `num_rendered`.
- Added `CudaRasterizer binning buffer bytes` logging before binning-buffer allocation.
- Changed CUDA buffer resizing in `GaussianView.cpp` to use checked allocation/free calls.
- Added braces around the checked `cudaFree` path to keep macro expansion unambiguous.
- Updated `code/scripts/run_sibr_gaussian_viewer.sh` to default to `--rendering-size 960 540`, reducing tile pressure on GTX 1080 Ti / compute capability 6.1.
- Allowed `SIBR_RENDER_WIDTH` and `SIBR_RENDER_HEIGHT` environment variables to override the wrapper render size for quick low-resolution diagnostics.

Rebuilt and reinstalled SIBR:

```bash
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

Verification:

- Build/install succeeded.
- Installed `libsibr_gaussian.so` contains:
  - `CudaRasterizer forward stats: P=`
  - `CudaRasterizer binning buffer bytes=`
  - `Disabling fast Gaussian culling by default on compute capability 6.x for compatibility.`

Next run should report the new stats lines before any remaining crash. If `num_rendered` or binning-buffer size is excessive, the next fix should clamp/diagnose tile expansion. If those values are sane but `duplicate_with_keys` still crashes, add a bounded write guard and overflow flag to the duplication kernel for compute capability 6.x diagnostics.

## Follow-Up: Binning Buffer Size Overflow

The user's next run produced:

```text
CudaRasterizer forward stats: P=123060, resolution=960x631, tile_grid=60x40, num_rendered=69984585
CudaRasterizer binning buffer bytes=-1756841089
out of memory
```

This identified a host-side size overflow before the CUDA duplication kernel could be tested cleanly. `required<BinningState>(num_rendered)` returns `size_t`, but the local rasterizer code stored it in `int binning_chunk_size`. For this scene, the required binning allocation is larger than 2 GiB, so the signed 32-bit value wrapped negative and then propagated into allocation.

Fixes:

- Changed `img_chunk_size` and `binning_chunk_size` in `rasterizer_impl.cu` from `int` to `size_t`.
- Read `num_rendered` as `uint32_t`, promote to `size_t` for buffer sizing, and cast back to `int` only after an explicit range check for kernel/CUB calls.
- Returned `num_rendered_int` from the forward pass to match the existing `int` API.
- Changed `resizeFunctional` so large CUDA buffer requests (`>=512 MiB`) allocate exactly the requested size instead of always over-allocating by 2x.

Rebuilt and reinstalled SIBR:

```bash
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

Verification:

- Build/install succeeded.
- Installed `libsibr_gaussian.so` contains the new range-check string:

```text
CudaRasterizer forward requires num_rendered to fit in int for kernel launches.
```

Next:

- Ask the user to rerun `code/scripts/run_sibr_gaussian_viewer.sh`.
- Expected improvement: `CudaRasterizer binning buffer bytes` should now be a positive `size_t` value instead of a negative overflow.
- If the viewer still reports `out of memory`, retry with `SIBR_RENDER_WIDTH=640 SIBR_RENDER_HEIGHT=360`.
- If allocation succeeds but `duplicate_with_keys` fails again, proceed to a guarded duplication-kernel diagnostic.

## Follow-Up: Viewer Opens But Object Is Not Visible

After the overflow fix, the user reported that the SIBR window opens and mouse interaction changes colors, but the object is not clearly visible in the `Point view`.

Diagnostics:

- GOF prediction images under `train/ours_1000/test_preds_8` are non-empty, so the trained model is not blank.
- Raw COLMAP camera centers and model `cameras.json` are in the same normalized scale, roughly within `[-6, 5]`.
- The Gaussian PLY has a small number of extreme outliers:
  - position radius max: about `253`
  - `max_scale` max: about `63.8`
  - `max_scale > 10`: `59` Gaussians
  - `radius < 10`: `122475 / 123060` Gaussians
- These outliers can dominate the interactive SIBR view and inflate tile workloads even though most Gaussians are near the scene.

Changes:

- Added `wind3dgs.m04_mesh_extraction.filter_viewer_safe_ply`, a binary PLY filtering utility that preserves the original 3DGS record layout and drops only extreme visualization outliers.
- Generated:

```text
experiments/M04_mesh_extraction/models/gof_playroom_i1000_r8/point_cloud/iteration_viewer_safe/point_cloud.ply
```

- Filtering result:
  - input vertices: `123060`
  - output vertices: `122612`
  - dropped by radius: `389`
  - dropped by scale: `59`
- Added `SIBR_DEFAULT_ITERATION` wrapper override so the filtered PLY can be launched without long positional arguments.
- Patched SIBR fallback copy renderer to update its width/height uniforms from the actual render target on every `process()` call.
- Rebuilt and reinstalled SIBR successfully.
- Documented the viewer-safe filtering workflow in `experiments/M04_mesh_extraction/README.md`.

Verification:

```bash
PYTHONPATH=code .venv/bin/python -m wind3dgs.m04_mesh_extraction.filter_viewer_safe_ply --help
code/scripts/run_sibr_gaussian_viewer.sh --help
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

Next command for the user:

```bash
SIBR_DEFAULT_ITERATION=viewer_safe code/scripts/run_sibr_gaussian_viewer.sh
```

If the filtered model still does not show, the next likely target is the fallback `copy.frag` path or SIBR initial camera/subview state rather than the PLY content.

## Follow-Up: Flat Colored Point View

The user showed that the SIBR UI is responsive and the `Point view` subview is being drawn, but the subview is filled with a mostly flat blue/green color and no recognizable object.

Observations:

- `cfg_args` has `white_background=False`, so a purely empty render should be black, not blue/green.
- CudaRasterizer writes planar RGB as `out_color[ch * H * W + pix_id]`, matching the fallback `copy.frag` shader layout.
- This leaves two likely causes:
  - CudaRasterizer is producing a nearly flat non-black buffer for the current camera.
  - The fallback SSBO copy path is displaying the CUDA buffer incorrectly.

Changes:

- Added first-three-frame fallback buffer stats in `GaussianView.cpp` after the CUDA-to-CPU roundtrip:

```text
SIBR fallback frame <n> stats: resolution=... R[min,max,mean]=... G[min,max,mean]=... B[min,max,mean]=... centerRGB=...
```

- Added optional first-frame PPM dump controlled by:

```bash
SIBR_FALLBACK_DUMP_PPM=/tmp/sibr_fallback.ppm
```

- Rebuilt and reinstalled SIBR successfully.

Next command for diagnosis:

```bash
SIBR_DEFAULT_ITERATION=viewer_safe SIBR_FALLBACK_DUMP_PPM=/tmp/sibr_fallback.ppm code/scripts/run_sibr_gaussian_viewer.sh
```

Interpretation:

- If fallback stats show a varied image and the PPM looks correct, the issue is the OpenGL SSBO copy/display path.
- If fallback stats are nearly constant and the PPM is flat, the issue is the SIBR camera/rasterizer input state.
- In the UI, also try `Snap to closest` in the `Camera Point view` panel to reset to a nearby dataset camera.

## Follow-Up: GOF PLY Layout Mismatch Found

The user reran SIBR with fallback buffer stats. The viewer still produced a nearly flat image:

```text
SIBR fallback frame 0 stats: resolution=960x631 R[min,max,mean]=0.440729,0.471016,0.449519 G[min,max,mean]=0.499799,0.500052,0.49994 B[min,max,mean]=0.381317,0.624292,0.576353
```

This showed that the fallback CPU/OpenGL copy path was not the primary cause; the CUDA rasterizer output itself was already almost flat.

Inspection found a likely format mismatch:

- GOF output PLY has `63` float properties and stride `252` bytes.
- The final extra property is `filter_3D`.
- GraphDeco/SIBR's `loadPly()` reads the binary vertex payload directly into `RichPoint<3>`, which expects the standard `62` float properties and stride `248` bytes.
- Therefore, using the GOF PLY directly causes every record after the first one to be read 4 bytes out of phase.

Changes:

- Extended `wind3dgs.m04_mesh_extraction.filter_viewer_safe_ply` with `--sibr-compatible`.
- The option rewrites the output to GraphDeco/SIBR's exact 62-float property layout:
  - `x y z`
  - `nx ny nz`
  - `f_dc_0..2`
  - `f_rest_0..44`
  - `opacity`
  - `scale_0..2`
  - `rot_0..3`
- Generated:

```text
experiments/M04_mesh_extraction/models/gof_playroom_i1000_r8/point_cloud/iteration_sibr_safe/point_cloud.ply
```

Verification:

- Input vertices: `123060`
- Output vertices: `122612`
- Dropped outliers: `448`
- Dropped property: `filter_3D`
- Output property count: `62`
- Output stride: `248` bytes

Next command for the user:

```bash
SIBR_DEFAULT_ITERATION=sibr_safe code/scripts/run_sibr_gaussian_viewer.sh
```

Expected diagnostic change:

- `SIBR fallback frame` stats should become much more varied if the PLY layout mismatch was the root cause.
- The object should become visible in `Point view`.

Confirmed by user:

- Running `SIBR_DEFAULT_ITERATION=sibr_safe code/scripts/run_sibr_gaussian_viewer.sh` opened the SIBR viewer.
- The `Point view` now shows the playroom scene instead of a flat color buffer.
- This confirms that the main blank-view issue was the GOF `filter_3D` extra-property mismatch with SIBR's fixed binary `RichPoint<3>` loader.
- The current visual quality is blurry because this is the quick `1000`-iteration smoke model, not a quality-oriented 3DGS training run.

## Follow-Up: CudaRasterizer Forward Stage Diagnostics

The previous compatibility patches did not resolve the illegal memory access. The error still occurred after `CudaRasterizer::Rasterizer::forward`, which meant the exact internal CUDA stage was unknown.

Additional diagnostic patch:

- Added synchronous CUDA checks inside `rasterizer_impl.cu` after:
  - `forward/preprocess`
  - `forward/prefix_sum_tiles_touched`
  - `forward/read_num_rendered`
  - `forward/duplicate_with_keys`
  - `forward/radix_sort`
  - `forward/clear_tile_ranges`
  - `forward/identify_tile_ranges`
  - `forward/render_tiles`
- Each failure now throws a message like:

```text
CudaRasterizer forward failed after forward/<stage>: <cuda error>
```

Rebuilt and reinstalled SIBR:

```bash
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

Verification:

- Build/install succeeded.
- Installed `libsibr_gaussian.so` contains the new `forward/...` diagnostic strings.

Ask the user to rerun and report the new stage-specific CUDA error:

```bash
code/scripts/run_sibr_gaussian_viewer.sh
```

## Follow-Up: CC 6.1 CUDA Illegal Memory Access

After the compute-capability guard was relaxed, the viewer loaded:

- 225 input cameras
- 37,005 SfM points
- 123,060 Gaussian splats

Then rendering aborted with:

```text
A CUDA error occurred during rendering: an illegal memory access was encountered.
```

Additional compatibility/safety patches:

- Disabled fast Gaussian culling by default on compute capability `6.x`. The fast-culling path in the upstream rasterizer is marked as more aggressive and potentially in need of math cleanup.
- Changed the OpenGL image buffer from immutable `glNamedBufferStorage` to resizable `glNamedBufferData`.
- Added dynamic resizing for the CUDA fallback image buffer used by `--no_interop`.
- Added a synchronous CUDA check immediately after `CudaRasterizer::Rasterizer::forward` so future errors are reported closer to the offending render pass.

Rebuilt and reinstalled SIBR:

```bash
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

Verification:

- Build/install succeeded.
- Installed `libsibr_gaussian.so` contains:
  - `Disabling fast Gaussian culling by default on compute capability 6.x for compatibility.`
  - `Resizing CUDA fallback image buffer from ...`

Ask the user to rerun:

```bash
code/scripts/run_sibr_gaussian_viewer.sh
```

## Follow-Up: Compute Capability 6.1

After the WSLg window-position warning was downgraded, the viewer loaded the playroom cameras, SfM points, and GL mesh successfully, then aborted in `GaussianView.cpp`:

```text
Sorry, need at least compute capability 7.0+!
```

The local GPU is GTX 1080 Ti with compute capability 6.1. The CudaRasterizer target had already been patched and built with `sm_61`; the remaining blocker was an unconditional runtime guard in the SIBR Gaussian viewer constructor.

Additional patch:

- Changed the fatal guard from `prop.major < 7` to `prop.major < 6`.
- Added a warning for `6.x` devices explaining that upstream recommends 7.0+, but this local build includes `sm_61` for GTX 1080 Ti compatibility.

Rebuilt and reinstalled SIBR:

```bash
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

Verification:

- Build/install succeeded.
- Installed `libsibr_gaussian.so` contains the new compute capability warning and `sm_61` target strings.

Ask the user to rerun:

```bash
code/scripts/run_sibr_gaussian_viewer.sh
```

If it still aborts, the next output should include the concrete GLEW error string.

## Follow-Up: Wayland Window Position Callback

After the GLEW compatibility patch, the user reached the next WSLg/Wayland-specific issue:

```text
Wayland: The platform does not provide the window position
```

This is caused by SIBR calling `glfwSetWindowPos` during window initialization. Wayland does not let applications control global window position, so GLFW reports this as a platform warning/error through the callback. SIBR's callback treated every GLFW callback as fatal.

Additional patch:

- In `Window.cpp`, GLFW callback messages starting with `Wayland:` and containing `window position` are now logged as warnings and allowed to continue.
- Other GLFW errors still abort.

Rebuilt and reinstalled SIBR again with:

```bash
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
```

Verification:

- Build/install succeeded.
- Installed `libsibr_graphics.so` contains the new Wayland/window-position handling strings.

Ask the user to rerun:

```bash
code/scripts/run_sibr_gaussian_viewer.sh
```
