# LIO-SAM-GPU-ScanToMapOpt
This repository reimplements the line/plane odometry (based on LOAM) of LIO-SAM with CUDA.  Replacing pcl's kdtree, a point cloud hash map (inspired by iVox of [Faster-LIO](https://github.com/gaoxiang12/faster-lio)) on GPU is used to accelerate local map building, 5-neighbour KNN search and non-linear optimization.

Modifications are as follow : 
- The CUDA codes of the line/plane odometry are in [src/cuda_plane_line_odometry](https://github.com/qdLMF/LIO-SAM-CUDA-ScanToMapOpt/tree/master/src/cuda_plane_line_odometry). 
- To use this CUDA odometry, the scan2MapOptimization() in mapOptimization.cpp is replaced with scan2MapOptimizationWithCUDA().


## About
This repository reimplements the line/plane odometry in scan2MapOptimization() of mapOptimization.cpp with CUDA. 

On my machine (Orin-NX-8GB, walking_dataset.bag, with OpenMP), original CPU version: 
- average cost of extracting surrounding scans is more than 30ms
- average cost of building local map is about 20ms
- average cost of KNN search and optimization is about 30ms
- average cost of all operations in one frame is about 85ms

This repository replaces pcl's kdtree with a point cloud hash map (inspired by iVox of [Faster-LIO](https://github.com/gaoxiang12/faster-lio)) implemented with CUDA. 

Meanwhile, other parts of the line/plane odometry (jacobians & residuals etc) are also implemented with CUDA. 

On my machine (Orin-NX-8GB, walking_dataset.bag), GPU version implemented by this project :
- average cost of extracting surrounding scans is down to about 2.74ms
- average cost of incrementally updating local map is down to about 1.16ms
- average cost of one 5-neighbour KNN search is down to about 1.40ms
- average cost of all operations in one frame is down to about 21.56ms


## Dependencies
The essential dependencies are as same as [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)

My Orin-NX-8GB's specific enviroment :
- Ubuntu 20.04, Ros Noetic, JetPack 5.10
- C++14
- [CUDA](https://developer.nvidia.com/cuda-downloads) 11.4
- [Eigen](https://eigen.tuxfamily.org/) 3.3.7


# How To Build
Before build this repo, some CMAKE variables in [src/cuda_plane_line_odometry/CMakeLists.txt](https://github.com/qdLMF/LIO-SAM-GPU-ScanToMapOpt/blob/master/src/cuda_plane_line_odometry/CMakeLists.txt) need to be modified to fit your enviroment : 
```
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)   # change it to your path to nvcc
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/bin/nvcc) # change it to your path to nvcc
set(CMAKE_CUDA_ARCHITECTURES 87)                    # for example, if your device's compute capability is 6.2, then set this CMAKE variable to 62
                                                    # In my Orin-NX-8GB, this CMAKE variable is 87 
```
The basic steps to compile and run this repo is as same as [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM).


## Speed-up
<table style="text-align:center;">

<tr>
<th rowspan="2">Sequence</th><th colspan="3">Orin-NX-8GB CPU</th><th colspan="5">Orin-NX-8GB GPU</th>
</tr>

<tr>
<th>extract<br>surrounding<br>scans</th><th>build<br>kdtree</th><th>one<br>frame</th><th>extract<br>surrounding<br>scans</th><th>update<br>hashmap</th><th>one<br>KNN</th><th>one<br>frame</th><th>speed-up</th>
</tr>

<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Walking</a></td><td>34.65ms</td><td>20.03ms</td><td>84.95ms</td><td>2.74ms</td><td>1.16ms</td><td>1.40ms</td><td>21.56ms</td><td>3.94x</td>
</tr>

<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Campus (large)</a></td><td>25.21ms</td><td>19.34ms</td><td>84.75ms</td><td>1.71ms</td><td>1.13ms</td><td>1.49ms</td><td>23.58ms</td><td>3.59x</td>
</tr>

<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">2011_09_30_drive_0028</a></td><td>68.17ms</td><td>22.04ms</td><td>166.67ms</td><td>11.70ms</td><td>3.97ms</td><td>2.59ms</td><td>54.06ms</td><td>3.08x</td>
</tr>

<!--
<tr>
<td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td>
</tr>
-->
</table>



## Acknowledgements
This repository is a modified version of [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM), whose line/plane odometry is originally based upon LOAM.

The point cloud hash map on GPU is inspired by iVox data structure of [Faster-LIO](https://github.com/gaoxiang12/faster-lio), and draws experience from [kdtree_cuda_builder.h](https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/kdtree_cuda_builder.h) of [FLANN](https://github.com/flann-lib/flann).


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=qdLMF/LIO-SAM-GPU-ScanToMapOpt&type=Timeline)](https://star-history.com/#qdLMF/LIO-SAM-GPU-ScanToMapOpt&Timeline)
