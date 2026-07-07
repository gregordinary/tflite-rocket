# Frigate with the RK3588 NPU (rocket) detector.
#
# Stage the build artifacts into this directory first (see README.md "Build and
# run"); all three must be present in the build context:
#   libtflite_rocket.so      — the delegate (build with -DTFLITE_ROCKET_PORTABLE_GLIBC=ON
#                              so it loads on the image's glibc; see README)
#   libtflite_cshim.so       — provides the TFLite C-API symbols the headers-only
#                              delegate binds at dlopen (LD_PRELOADed via the compose)
#   rocket_mobiledet.tflite  — the SSDLite-MobileDet model
#
# Default to the stock (standard) Frigate image. The rocket detector is image-agnostic;
# override with --build-arg FRIGATE_VERSION=0.17.2-rk for the RK3588 build that bundles
# rkmpp hardware video decode (useful for several real cameras).
ARG FRIGATE_VERSION=0.17.2
FROM ghcr.io/blakeblackshear/frigate:${FRIGATE_VERSION}

# The detector plugin — Frigate's plugin loader auto-registers a "rocket" detector
# from any module in this package at startup, so no other Frigate change is needed.
COPY rocket.py /opt/frigate/frigate/detectors/plugins/rocket.py

# The self-contained delegate: librocketnpu is statically linked in and it reaches
# /dev/accel via raw ioctls, so this one file is the whole runtime.
COPY libtflite_rocket.so /usr/local/lib/libtflite_rocket.so

# The delegate is a classic C TfLiteDelegate built against the TFLite C-API headers
# alone, so it binds TfLiteIntArrayCreate/Free at dlopen. tflite_runtime loads its C
# extension RTLD_LOCAL, so those symbols aren't globally visible — this tiny shim
# provides them and is LD_PRELOADed (see docker-compose.yml).
COPY libtflite_cshim.so /usr/local/lib/libtflite_cshim.so

# The detector model (SSDLite-MobileDet uint8, validated at COCO mAP = CPU parity).
COPY rocket_mobiledet.tflite /models/rocket_mobiledet.tflite
