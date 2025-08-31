import pycuda.driver as cuda
cuda.init()  # Initialize CUDA driver
num_devices = cuda.Device.count()
print(f"Number of CUDA devices: {num_devices}")
