"""Shared admission budget for CountGD++ inference, without importing CUDA.

RTX 4090 measurements at the configured 800/1333 resize ceiling reached
2303 MiB allocated / 2690 MiB reserved. Four GiB leaves room for the CUDA
context and transient allocations; it is a resource guard, not model size.
See outputs/orange_pack_rods_20261018/gpu_measurements*.json.
"""

COUNTGDPP_MIN_FREE_MIB = 4096
COUNTGDPP_MIN_FREE_BYTES = COUNTGDPP_MIN_FREE_MIB * 1024**2

# Keep both sessions loaded for this dedicated counting service. Explicit
# release, worker failure/timeout and server shutdown still release resources.
COUNTGDPP_IDLE_TIMEOUT_SECONDS = None
SCENE_IDLE_TIMEOUT_SECONDS = None
