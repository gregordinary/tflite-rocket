# Frigate on the RK3588 NPU

Run [Frigate](https://frigate.video) object detection on the Rockchip RK3588 NPU through the
tflite-rocket delegate. The detector's convolutional backbone runs on the NPU, and the rest of
Frigate is unchanged.

This directory is a complete deployment:

- A detector plugin.
- A Docker image that adds it to stock Frigate.
- A Compose file.
- A configuration that brings the stack up on a looped sample clip, so you can confirm detections
  before pointing a real camera at it.

## Contents

| File | Purpose |
|---|---|
| `rocket.py` | The Frigate detector plugin. It registers a `rocket` detector that loads `libtflite_rocket.so`. It is a thin adapter over an SSD `.tflite`, and its detection output is byte-for-byte Frigate's standard SSD path. |
| `Dockerfile` | Adds `rocket.py`, the delegate `.so`, the C-API shim `libtflite_cshim.so` and the model to the Frigate image. |
| `docker-compose.yml` | Deploys the image with the NPU device and the host `render` group. |
| `config.yml` | A `rocket` detector and a looped sample clip for bring-up. |
| `make-sample-clip.sh` | Writes `media/sample.mp4`: a still period, then a person crosses (see [Troubleshooting](#no-detections-on-the-sample-clip)). |

## Detector path

Frigate compiles its detectors as Python plugins that it discovers at startup. It includes an
`rknn` detector and no `rocket` one. `rocket.py` is that plugin. It loads an unmodified
SSDLite-MobileDet `.tflite` with the tflite-rocket external delegate through `load_delegate`, and
the delegate runs the convolutional backbone on the NPU. The SSD `TFLite_Detection_PostProcess`
op is a custom op and stays on the CPU, as it does for Frigate's built-in `cpu` and `edgetpu`
detectors. The detection output and the labelmap are therefore identical to those detectors'.

The `librocketnpu` driver library is linked into the delegate statically, and it opens
`/dev/accel/accel0` with raw ioctls. The container therefore needs the delegate and the shim, the
device node and the host `render` group. The build links libdrm, and a toolchain that links with
`--as-needed` drops it from the `.so`'s dependencies. Check which libraries the `.so` needs with
`readelf -d libtflite_rocket.so | grep NEEDED`.

The model is SSDLite-MobileDet (uint8, 320×320), Frigate's default CPU SSD model. Under the
delegate's current defaults, its COCO mAP@[.5:.95] over 500 val2017 images is 0.3325 against the
CPU interpreter's 0.3301 [HW sweep 2026-10-29].

## Requirements

- An RK3588 board on a mainline kernel with the `rocket` DRM-accel driver and
  `/dev/accel/accel0`.
- The delegate and `librocketnpu` built on that host, per the top-level
  [README](../README.md#build-and-run). The C++ delegate needs `_Float16` as an extended
  floating-point type, which takes GCC 13 or later on Arm64. Build it on the host rather than in
  a bookworm container, whose GCC 12 accepts it only on x86.
- Docker with Compose.
- The model, `ssdlite_mobiledet_coco_qat_postprocess.tflite`, from the Coral or MediaPipe model
  zoo.

The `.so` is copied into the Frigate image, which is Debian 12 bookworm with glibc 2.36. A newer
host links symbols the image lacks: glibc 2.38's `__isoc23_*`, and on a glibc 2.43 host `sqrtf`.
The `-DTFLITE_ROCKET_PORTABLE_GLIBC=ON` build option links forwarders for them into the `.so`,
which then needs only glibc 2.34 and loads on bookworm. If a symbol still needs a newer glibc, or
a newer libstdc++ than bookworm's, the build fails.

## Build and run

```bash
# 1. Build the delegate and the C-API shim on the RK3588 host, portable to the image's glibc.
#    librocketnpu is already built and installed at /usr/local, per ../README.md.
cmake -S .. -B ../build -DTFLITE_DIR=/path/to/tflite/headers \
      -DTFLITE_ROCKET_PORTABLE_GLIBC=ON
cmake --build ../build -j -t tflite_rocket tflite_cshim

# 2. Stage the two .so files and the model into this directory, the build context.
cp ../build/libtflite_rocket.so ../build/libtflite_cshim.so .
cp /path/to/ssdlite_mobiledet_coco_qat_postprocess.tflite rocket_mobiledet.tflite

# 3. Write a sample clip for bring-up: a still period while motion calibration finishes, then a
#    person crosses (see Troubleshooting). Or put your own clip at media/sample.mp4.
./make-sample-clip.sh

# 4. Confirm that the render gid in docker-compose.yml (group_add) matches /dev/accel/accel0.
getent group render        # e.g. render:x:988:...  -> group_add: ["988"]

# 5. Build and start.
docker compose build
docker compose up -d
docker compose logs -f frigate
```

The shim, `libtflite_cshim.so`, provides the two TFLite C-API symbols the delegate binds at
`dlopen`, and the Compose file preloads it with `LD_PRELOAD`. The image's `tflite_runtime` loads
its C extension `RTLD_LOCAL`, so the delegate cannot see that module's copies of the symbols.

The Frigate UI comes up on `https://<host>:8971`. On the System -> Detectors page the `rocket`
detector reports its inference speed, and the Debug view draws live boxes on the sample clip.

## Detector configuration

```yaml
detectors:
  rocket:
    type: rocket
    num_threads: 4        # NPU worker fan-out across the 3 cores (1..8)
    # device: "4"         # pin this detector's worker to A76 core 4 (see Multi-camera)
    # native_int8: true   # native int8/uint8 conv path (default; mAP-validated)
    # delegate_path: /usr/local/lib/libtflite_rocket.so

model:
  path: /models/rocket_mobiledet.tflite
  width: 320
  height: 320
  input_tensor: nhwc
  input_pixel_format: rgb
  input_dtype: int
  model_type: ssd
```

| Option | Default | Effect |
|---|---|---|
| `num_threads` | `4` | Worker fan-out across the three NPU cores. The delegate clamps it to 1-8, and 4 is fastest for one detector. |
| `device` | unset | The Cortex-A76 core (`"4"` to `"7"`) that this detector's worker pins to. It sets `ROCKET_CPU_AFFINITY` for the detector process. |
| `native_int8` | `true` | Runs the quantized convolutions on the delegate's native int8 routes, at CPU parity on COCO mAP. |
| `delegate_path` | `/usr/local/lib/libtflite_rocket.so` | The delegate's location in the container. |

## Multi-camera throughput

One warm MobileDet stream does not reach video rate. Throughput comes from running several
detectors at once, one per A76 core. Frigate runs one process per configured detector, so pin
each to a distinct core:

```yaml
detectors:
  rocket0: { type: rocket, num_threads: 1, device: "4" }
  rocket1: { type: rocket, num_threads: 1, device: "5" }
  rocket2: { type: rocket, num_threads: 1, device: "6" }
  rocket3: { type: rocket, num_threads: 1, device: "7" }
```

A pool of these four detectors served four cameras:

| Detectors | Aggregate `detection_fps` | Scaling |
|---:|---:|---:|
| 1 | 3.20 | 1.00x |
| 2 | 6.00 | 1.88x |
| 3 | 8.00 | 2.50x |
| 4 | 9.55 | 2.98x |

**Conditions:** RK3588 (Turing RK1), NPU at ~816 MHz (the 600 MHz request), NPU IRQs pinned to
the big cores, measured live on 2026-07-01. A single inference took ~336 ms on that build.

**Not characterized:** the live pool at the delegate's current 51.4 ms single stream. Outside
Frigate, four delegate processes on SSDLite-MobileDet, one per A76 core, measured 3.33-3.51x one
process's rate [HW sweep 2026-10-29]. The NPU ran at 928 MHz (the 900 MHz request).

Each detector ran on its own A76 core, the four at ~73% busy, while the Cortex-A55 cores handled
video decode. Without Frigate, the delegate on a test unit whose cost is the submit scales to
3.56x at four processes (`../tools/pool_throughput.py`). The live pool scaled less, because that
build's inference was bound by the host's cube packing and gathering rather than by the submit.
Per-inference latency rose from 338 to 424 ms as the four contexts contended for DRAM bandwidth.

For the best result, pin the NPU IRQs to the big cores with `rocket-userspace`'s
`tools/npu_set_irq_affinity.sh throughput`.

Frigate's System-page `inference_speed` measured ~250 ms on that build. It is Frigate's own
per-invoke timer, which covers less than the ~336-338 ms full single-stream inference.

## Using a real camera

This deployment runs unchanged against live RTSP cameras, and only the `cameras` block changes.
On 2026-07-01 the `rocket` detector detected `person` and `car` on live cameras on the NPU. Its
`inference_speed` matched the sample clip's, ~250 ms on that build.

Replace the sample `cameras` block with your camera, restreamed through go2rtc. Drop its
`motion:` override, since a real camera calibrates on its own still periods. The RK3588's
hardware video decoder (`preset-rkmpp`) keeps the CPU free for detection. It is in the `-rk`
image, so build with `--build-arg FRIGATE_VERSION=0.17.2-rk` to use it, or drop the
`hwaccel_args` line for software decode on the standard image:

```yaml
go2rtc:
  streams:
    front: ["rtsp://<user>:<password>@<camera-ip>:554/stream1"]

cameras:
  front:
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8554/front
          input_args: preset-rtsp-restream
          roles: [detect]
      hwaccel_args: preset-rkmpp # -rk image only; remove for software decode
    detect:
      width: 1280
      height: 720
      fps: 5
```

## Troubleshooting

### `Failed to load the rocket delegate`

The delegate's `.so` did not load, and the message after the path says why. The plugin does not
open the NPU at this point, so the cause is the file or its dependencies:

- **`undefined symbol: TfLiteIntArrayCreate`.** The shim is not preloaded. Confirm that
  `libtflite_cshim.so` is in the image and that `LD_PRELOAD` names it, as the Compose file does.
- **`version 'GLIBC_2.38' not found`** (or `GLIBC_2.43`). The `.so` was built on a host newer than
  the Frigate image. Rebuild it with `-DTFLITE_ROCKET_PORTABLE_GLIBC=ON`, then stage it again and
  rebuild the image.
- **No such file.** The `.so` is not at `delegate_path` in the container.

### Detections run slowly, on the CPU

The System page shows a slow inference, and the NPU is idle. Two causes have this symptom:

- **No NPU device in the container.** The delegate then computes on the CPU and logs `an on-NPU
  path was requested but no NPU device opened`. Confirm the `devices:` mapping and that
  `group_add:` matches the host `render` gid (`getent group render`). The node is
  `crw-rw---- root render`, so the container process needs that group.
- **XNNPACK claimed the convolutions first.** The plugin prevents this with
  `BUILTIN_WITHOUT_DEFAULT_DELEGATES`. If you adapted it, keep that resolver.

### Slow first inference

The NPU clock parks at idle, so the first inferences measure slow. Steady state follows after a
few.

### The admin password or the database resets on `docker compose down`

The Compose file mounts `config.yml` alone into `/config`. Frigate keeps its database, its
authentication and `model_cache` in `/config`, inside the container, so they are lost each time the
container is recreated. A new one-time admin password then prints on each start. To keep them, mount a host
directory at `/config` and keep `config.yml` inside it.

### No detections on the sample clip

`detection_fps` stays at 0 because Frigate's motion detector starts by calibrating, and it ignores
all motion until motion falls below ~5% of the frame. A clip that moves continuously, such as a
pan or an object always on screen, never finishes calibrating. The detector then never runs, not
even Frigate's own CPU detector.

The clip needs a few still seconds first, then movement, which is what `make-sample-clip.sh`
writes. A real camera gets still periods naturally.

Check progress with `docker exec frigate curl -s localhost:5000/api/stats`. The `rocket` detector
reports a nonzero `inference_speed` once it runs on the NPU, ~250 ms on the 2026-07-01 build.

### No events from a live camera

The detector runs, and no events appear for long stretches. Frigate runs detection only on motion
regions, and creates events only for the object types in the camera's `track` list. On a quiet
outdoor scene, swaying foliage is often the only motion. The detector runs on those regions and
finds nothing trackable, so the System page shows a live `inference_speed` while `detection_fps`
and events stay near zero.

The sample clip confirms the NPU path, because it has a scripted person. Two points apply to a
live scene:

- The SSD-MobileDet model runs at 320×320, so a distant or small object can fail to clear the
  score threshold. Frame the camera so objects of interest fill a reasonable part of the view.
- A stationary object the model reads as a class outside the `track` list is filtered out, so it
  never becomes an event. A white wall reads as `boat`, and a window as `clock`.
