import os
import sys
import platform
import jetson_utils
# import cuda
import vpi
import cupy as cp
import pycuda as cuda

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"CuPy version: {cp.__version__}")
print(f"VPI version: {vpi.__version__}")
# print(f"Jetson Utils version: {jetson_utils.__version__}")

# Check CUDA version reported by CuPy
try:
    # import cuda
    print(f"CUDA Driver Version: {cuda.runtime.driverGetVersion()}")
    print(f"CUDA Runtime Version: {cuda.runtime.runtimeGetVersion()}")
except ImportError:
    print("NVIDIA CUDA Python package not found, cannot get driver/runtime versions easily.")
    print("Check `nvcc --version` or `/usr/local/cuda/bin/nvcc --version`")

# Check environment variables
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"PATH (relevant parts): {':'.join([p for p in os.environ.get('PATH', '').split(':') if 'cuda' in p or 'bin' in p or 'local' in p])}")
print(f"LD_LIBRARY_PATH (relevant parts): {':'.join([p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if 'cuda' in p or 'lib' in p or 'local' in p])}")

# Small test to see if CuPy can use CUDA Array Interface
try:
    a = cp.arange(10).reshape(2,5)
    b = cp.asarray(a) # This implicitly uses __cuda_array_interface__ for copy
    print(f"CuPy test: a has __cuda_array_interface__: {hasattr(a, '__cuda_array_interface__')}")
    print(f"CuPy test: b created from a: {b.shape}")
except Exception as e:
    print(f"CuPy test failed: {e}")