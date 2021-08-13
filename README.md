# GPU_Check
A small script allowing to check if GPU is available (or not well configured) and compute some basic matrix multiplication to compare processing time using the CPU and the GPU.

## How to run ?
```
python Check_GPU_available.py 
```
If your Tensorflow GPU installation is isolated in a dedicated Conda environment, make sure you run this environment is activated before runing this script. 

## Output message
```
(tf_gpu) tais@tais-HP-Z620-Workstation:/media/tais/data/Dropbox/Linux/Programs_installation/NVIDIA_Tensorflow$ python Check_GPU_available.py 
Using TensorFlow backend.
2021-08-13 11:06:07.327345: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2021-08-13 11:06:07.388748: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:05:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
2021-08-13 11:06:07.390921: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2021-08-13 11:06:07.432355: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2021-08-13 11:06:07.453950: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2021-08-13 11:06:07.459327: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2021-08-13 11:06:07.499248: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2021-08-13 11:06:07.504667: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2021-08-13 11:06:07.577164: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-08-13 11:06:07.578276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2021-08-13 11:06:07.578907: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
2021-08-13 11:06:07.615934: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2693600000 Hz
2021-08-13 11:06:07.617634: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ba6085c480 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-08-13 11:06:07.617653: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-08-13 11:06:07.619518: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:05:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
2021-08-13 11:06:07.619576: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2021-08-13 11:06:07.619599: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2021-08-13 11:06:07.619618: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2021-08-13 11:06:07.619650: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2021-08-13 11:06:07.619669: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2021-08-13 11:06:07.619708: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2021-08-13 11:06:07.619732: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-08-13 11:06:07.620721: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2021-08-13 11:06:07.621241: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2021-08-13 11:06:07.787584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-08-13 11:06:07.787630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
2021-08-13 11:06:07.787641: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
2021-08-13 11:06:07.789699: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7255 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070 SUPER, pci bus id: 0000:05:00.0, compute capability: 7.5)
2021-08-13 11:06:07.794812: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ba619ddbc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-08-13 11:06:07.794876: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2070 SUPER, Compute Capability 7.5
1 Physical GPUs, 1 Logical GPUs
2021-08-13 11:06:07.802516: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:05:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
2021-08-13 11:06:07.802652: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2021-08-13 11:06:07.802694: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2021-08-13 11:06:07.802730: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2021-08-13 11:06:07.802792: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2021-08-13 11:06:07.802862: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2021-08-13 11:06:07.802919: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2021-08-13 11:06:07.802955: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-08-13 11:06:07.804737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
2021-08-13 11:06:07.804766: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-08-13 11:06:07.804775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
2021-08-13 11:06:07.804783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
2021-08-13 11:06:07.805799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/device:GPU:0 with 7255 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070 SUPER, pci bus id: 0000:05:00.0, compute capability: 7.5)
2021-08-13 11:06:09.596529: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-08-13 11:06:11.136016: W tensorflow/stream_executor/gpu/redzone_allocator.cc:312] Not found: ./bin/ptxas not found
Relying on driver to perform ptx compilation. This message will be only logged once.
2021-08-13 11:06:11.276154: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images (batch x height x width x channel). Sum of ten runs.
CPU (s):
0.7851287240000033
GPU (s):
0.0743922960000134
GPU speedup over CPU: 10x
```
Most of the output if not very interesting, except in case the GPU is not found. In this situation you should try to find some potential reason in the (error) message output. 

In the last 6 lines, you can see the benchmark results between CPU and GPU with a convolution of 32x7x7x3 filter over random 100x100x100x3 images (batch x height x width x channel). **In this exemple, the GPU processing is about ~10 times faster than CPU processing**. 

### Prerequisite
Of course, you need a proper installation of: 
- NVIDIA driver and compatible CUDA library. 
- Tensorflow-gpu (the GPU version of tensorflow)

A good check is to run the nvidia utility using this code `nvidia-smi` or `watch -n0.1 nvidia-smi`. If there is a mismatch between the version of the Nvidia driver and Cuda library, the utility will tell you imadiately. 
