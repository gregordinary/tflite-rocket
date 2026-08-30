# Frigate on the RK3588 NPU

Run [Frigate](https://frigate.video) object detection on the Rockchip RK3588 NPU
through the tflite-rocket delegate. The detector backbone runs on the NPU, and the
rest of Frigate is unchanged.

This directory is a complete deployment:

- A detector plugin.
- A Docker image that bakes it into stock Frigate.
- A compose file.
- A config that brings the stack up on a looped sample clip, so you can confirm
  detections before pointing a real camera at it.

## Contents

| File | Purpose |
|---|---|
| `rocket.py` | The Frigate detector plugin. Auto-registers a `rocket` detector that loads `libtflite_rocket.so`. A thin adapter over an SSD `.tflite`; detection output is byte-for-byte Frigate's standard SSD path. |
| `Dockerfile` | Bakes `rocket.py`, the delegate `.so`, and the model into the Frigate image. |
| `docker-compose.yml` | Deploys the image with the NPU device + render group wired in. |
| `config.yml` | A `rocket` detector + a looped sample clip for bring-up. |
| `make-sample-clip.sh` | Generates `media/sample.mp4` (a still period then a person crosses; see Troubleshooting). |

## Detector path

Frigate compiles its detectors as Python plugins discovered at startup (it ships a
`rknn` detector but no `rocket` one). `rocket.py` is that plugin: it
loads an unmodified SSDLite-MobileDet `.tflite` with the tflite-rocket external
delegate via `tflite.load_delegate`, which runs the convolutional backbone on the
NPU. The SSD `TFLite_Detection_PostProcess` op is a custom op and stays on the CPU,
exactly as it does for Frigate's built-in `cpu` and `edgetpu` detectors. The detection
output and the labelmap are therefore identical to those detectors.

The delegate `.so` is self-contained. The driver is statically linked in and it
talks to `/dev/accel/accel0` through raw ioctls. The container therefore needs only
that one file, plus the device node and the host `render` group. No libdrm, no extra
runtime.

**Model.** SSDLite-MobileDet (uint8, 320×320), Frigate's default CPU SSD model. On
the NPU it runs at COCO mAP parity with the CPU TFLite reference (0.3321 vs 0.3318
over 500 val2017 images), so detections match what the `cpu` detector would produce.

## Build and run

The delegate links the driver, so it is built on the RK3588 host, meaning a mainline
kernel with the `rocket` accel driver and `/dev/accel/accel0`. That is the standard
tflite-rocket build, in the top-level [README](../README.md).

Build it on the host rather than in a bookworm container. The C++ delegate needs
`_Float16` as a real extended-float type,
which is GCC 13+ (the host toolchain), not bookworm's GCC 12.

**One wrinkle: glibc.** The `.so` is copied into the Frigate image (Debian 12
bookworm, glibc 2.36). A newer host (DietPi/Debian trixie/forky) otherwise links two
glibc-2.38 `__isoc23_*` symbols the image lacks. The `-DTFLITE_ROCKET_PORTABLE_GLIBC=ON`
build option internalizes them so the `.so` floors at glibc 2.34 and loads on bookworm.

```bash
# 1. Build the delegate + the C-API shim on the RK3588 host, portable to the image's
#    glibc. (Driver already built+installed per ../README.md: librk3588npu at /usr/local.)
cmake -S .. -B ../build -DTFLITE_DIR=/path/to/tflite/headers \
      -DTFLITE_ROCKET_PORTABLE_GLIBC=ON
cmake --build ../build -j -t tflite_rocket tflite_cshim

# 2. Stage the two .so's + the model into this directory (the build context).
cp ../build/libtflite_rocket.so ../build/libtflite_cshim.so .
cp /path/to/ssdlite_mobiledet_coco_qat_postprocess.tflite rocket_mobiledet.tflite

# 3. Provide a sample clip for bring-up. ./make-sample-clip.sh writes one with the
#    right shape (a still period to let motion calibration finish, then a person
#    crosses); see Troubleshooting for why. Or drop in your own media/sample.mp4.
./make-sample-clip.sh

# 4. Confirm the host render gid in docker-compose.yml (group_add) matches /dev/accel0.
getent group render        # e.g. render:x:988:...  -> group_add: ["988"]

# 5. Build and start.
docker compose build
docker compose up -d
docker compose logs -f frigate
```

The shim (`libtflite_cshim.so`) provides the two TFLite C-API symbols the headers-only
delegate binds at `dlopen`. The compose `LD_PRELOAD`s it. tflite_runtime loads its own
C extension `RTLD_LOCAL`, so those symbols are not otherwise visible to the delegate.

The Frigate UI comes up on `https://<host>:8971`. On the System -> Detectors page
the `rocket` detector reports its inference speed, and the Debug view draws live
boxes on the sample clip.

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

| Option | Default | Meaning |
|---|---|---|
| `num_threads` | 4 | Worker fan-out across the 3 NPU cores. 4 is fastest for one detector. |
| `device` | unset | Big (A76) core (`"4"`..`"7"`) this detector's worker pins to. Sets `ROCKET_CPU_AFFINITY` for the detector process. |
| `native_int8` | true | Run quantized convs on the native int8 datapath (COCO-mAP-validated). |
| `delegate_path` | `/usr/local/lib/libtflite_rocket.so` | Delegate location in the container. |

## Multi-camera throughput

A single warm MobileDet inference is not video-rate on one stream. Throughput comes
from running several detectors concurrently, one per A76 core. Frigate runs one OS
process per configured detector, so the plugin spreads them across the big cores by
pinning each to a distinct core:

```yaml
detectors:
  rocket0: { type: rocket, num_threads: 1, device: "4" }
  rocket1: { type: rocket, num_threads: 1, device: "5" }
  rocket2: { type: rocket, num_threads: 1, device: "6" }
  rocket3: { type: rocket, num_threads: 1, device: "7" }
```

Measured live on the RK3588 at 600 MHz, with NPU IRQs pinned to the big cores and a
pool of `rocket0..rocket3` detectors serving four cameras:

| Detectors | Aggregate `detection_fps` | Scaling |
|---:|---:|---:|
| 1 | 3.20 | 1.00x |
| 2 | 6.00 | 1.88x |
| 3 | 8.00 | 2.50x |
| 4 | 9.55 | 2.98x |

Each detector lands on its own A76. The four big cores run ~73 % busy while the A55s
handle video decode.

A delegate-level submit-bound unit (`../tools/pool_throughput.py`) scales further, to
1.00x / 2.17x / 3.11x / 3.56x. The live pool tapers below that ceiling, because a real
MobileDet inference is host cube-scatter-and-gather-bound rather than submit-bound.
Per-inference latency therefore rises from 338 to 424 ms as the four contexts contend
for DRAM bandwidth. For the best result pin the
NPU IRQs to the big cores (`tools/npu_set_irq_affinity.sh throughput` in the driver repo).
Frigate's System-page `inference_speed` ~250 ms is its own per-invoke timer, a narrower span
than the ~336-338 ms full single-stream inference measured here.

## Using a real camera

This deployment runs unchanged against live RTSP cameras, and only the `cameras` block
changes. The rocket detector has been run end to end on live cameras. It detects
`person` and `car` on the NPU at the same `inference_speed ~250 ms` as the sample clip.

Replace the sample `cameras` block (and drop its `motion:` override, since a real camera
calibrates on its own still periods) with your camera, restreamed through go2rtc. The
RK3588's hardware video decoder (`preset-rkmpp`) keeps the CPU free for detection, and
it lives in the `-rk` image. So build with `--build-arg FRIGATE_VERSION=0.17.2-rk` to
use it, or drop the `hwaccel_args` line for software decode on the standard image:

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

- **Detector never loads / "Failed to load the rocket delegate".** The container
  can't open `/dev/accel/accel0`. Confirm the `devices:` mapping and that
  `group_add:` matches the host `render` gid (`getent group render`). The device is
  `crw-rw---- root render`, so the container process needs that group.
- **Detections run but the System page shows the inference on CPU / very slow.** The
  default TFLite delegate (XNNPACK) claimed the convs before the rocket delegate. The
  plugin guards against this with `BUILTIN_WITHOUT_DEFAULT_DELEGATES`. If you adapted
  it, keep that resolver.
- **`version 'GLIBC_2.38' not found` when the delegate loads.** The `.so` was built on
  a host newer than the Frigate image (Debian 12 bookworm). Rebuild it with
  `-DTFLITE_ROCKET_PORTABLE_GLIBC=ON` (internalizes the two glibc-2.38 `__isoc23_*`
  symbols), then re-stage and rebuild the image.
- **`undefined symbol: TfLiteIntArrayCreate` when the delegate loads.** The cshim isn't
  preloaded. Confirm `libtflite_cshim.so` is in the image and `LD_PRELOAD` points to it
  (the compose sets this).
- **First inference is slow.** The NPU clock parks at idle, so the first run reads low.
  Steady-state is reached after a few inferences.
- **The admin password or database resets on `docker compose down`.** The compose mounts
  only `config.yml`. Frigate's `/config` holds its DB, auth and `model_cache`. That lives
  inside the container and is lost on recreate, and a new one-time admin password prints
  on each start. To persist it, mount a host directory to `/config` and keep `config.yml`
  inside it.

### No detections on the sample clip, and `detection_fps` stays 0

Frigate's motion detector starts *calibrating* and ignores all motion until motion falls
below ~5% of the frame. A clip that moves continuously never finishes calibrating, such as
a pan or an object always on screen. The detector then never runs, not even Frigate's own
CPU detector.

The clip needs a few still seconds first, then movement, which is what
`make-sample-clip.sh` produces. A real camera gets still periods naturally.

Check progress with `docker exec frigate curl -s localhost:5000/api/stats`. The `rocket`
detector's `inference_speed` is ~250 ms when it is running on the NPU.

### A live camera runs the detector but logs no events for long stretches

Frigate only runs detection on motion regions, and only creates events for the object
types in the camera's `track` list. On a quiet outdoor scene, swaying foliage is often the
only motion. The detector fires on those regions and finds nothing trackable, so the
System page shows a live `inference_speed` while `detection_fps` and events stay near
zero.

That is motion and scene rather than a detector fault. The sample clip confirms the NPU
path, because it has a scripted person. Two practical points follow:

- The SSD-MobileDet model runs at 320×320, so a distant or small object can fail to clear
  the score threshold. Frame the camera so objects of interest are a reasonable fraction
  of the view.
- A stationary object the model misreads as a non-tracked class is correctly filtered out
  by the `track` list, so it never becomes an event. A white wall reads as `boat`, and a
  window as `clock`.
