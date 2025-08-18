# LIO-SAM-GPU-ScanToMapOpt
This repository reimplements the line/plane odometry (based on LOAM) of LIO-SAM with CUDA.  Replacing pcl's kdtree, a point cloud hash map (inspired by iVox of [Faster-LIO](https://github.com/gaoxiang12/faster-lio)) on GPU is used to accelerate 5-neighbour KNN search.

Modifications are as follow : 
- The CUDA codes of the line/plane odometry are in [src/cuda_plane_line_odometry](https://github.com/qdLMF/LIO-SAM-CUDA-ScanToMapOpt/tree/master/src/cuda_plane_line_odometry). 
- To use this CUDA odometry, the scan2MapOptimization() in mapOptimization.cpp is replaced with scan2MapOptimizationWithCUDA().


## About
This repository reimplements the line/plane odometry in scan2MapOptimization() of mapOptimization.cpp with CUDA. The most significant cost of the original implementation is the 5-neighbour KNN search using pcl's kdtree, which, on my machine (intel i7-6700k CPU, walking_dataset.bag, with OpenMP), usually takes about 5ms. This repository replaces pcl's kdtree with a point cloud hash map (inspired by iVox of [Faster-LIO](https://github.com/gaoxiang12/faster-lio)) implemented with CUDA. On my machine (Nvidia 980TI CPU, walking_dataset.bag), average cost of the 5-neighbour KNN search is down to about 0.5~0.6ms, average cost of all operations in one frame is down to about 11ms. Meanwhile, other parts of the line/plane odometry (jacobians & residuals etc) are also implemented with CUDA.


## Dependencies
The essential dependencies are as same as [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM), tested on Ubuntu 18.04 & Ros Melodic.

In addition, the CUDA reimplementation of the line/plane odometry requires : 
- C++14
- [CUDA](https://developer.nvidia.com/cuda-downloads) (>= 11.0)
- CUBLAS
- thrust
- [Eigen](https://eigen.tuxfamily.org/) (>= 3.3.9)


# How To Build
Before build this repo, some CMAKE variables in [src/cuda_plane_line_odometry/CMakeLists.txt](https://github.com/qdLMF/LIO-SAM-GPU-ScanToMapOpt/blob/master/src/cuda_plane_line_odometry/CMakeLists.txt) need to be modified to fit your enviroment : 
```
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)       # change it to your path to nvcc
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/bin/nvcc)   # change it to your path to nvcc
set(CMAKE_CUDA_ARCHITECTURES 52)                                        # for example, if your device's compute capability is 6.2, then set this CMAKE variable to 62
```
The basic steps to compile and run this repo is as same as [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM).


## Speed-up
<table style="text-align:center;">
<tr>
<th rowspan="2">Sequence</th><th colspan="2">CPU (Intel I7-6700K)</th><th colspan="3">GPU (Nvidia 980TI)</th>
</tr>
<tr>
<th>build kdtree</th><th>one frame<br>(build kdtree & all iteraions)</th><th>build hashmap</th><th>one KNN</th><th>one frame<br>(build hashmap & all iteraions)</th>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Walking</a></td><td>16.06ms no RVIZ<br>29.00ms with RVIZ</td><td>49.98ms no RVIZ<br>84.20ms with RVIZ</td><td>4.52ms no RVIZ<br>6.93ms with RVIZ</td><td>0.57ms no RVIZ<br>0.58ms with RVIZ</td><td>11.06ms no RVIZ<br>15.68ms with RVIZ</td>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Park</a></td><td>16.11ms no RVIZ<br>28.08ms with RVIZ</td><td>59.02ms no RVIZ<br>101.38ms with RVIZ</td><td>4.18ms no RVIZ<br>6.71ms with RVIZ</td><td>0.62ms no RVIZ<br>0.62ms with RVIZ</td><td>11.41ms no RVIZ<br>16.55ms with RVIZ</td>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Garden</a></td><td>17.66ms no RVIZ<br>31.71ms with RVIZ</td><td>53.40ms no RVIZ<br>84.24ms with RVIZ</td><td>5.01ms no RVIZ<br>7.43ms with RVIZ</td><td>0.60ms no RVIZ<br>0.61ms with RVIZ</td><td>11.42ms no RVIZ<br>15.66ms with RVIZ</td>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Rooftop</a></td><td>17.48ms no RVIZ<br>36.78ms with RVIZ</td><td>67.81ms no RVIZ<br>120.75ms with RVIZ</td><td>4.96ms no RVIZ<br>8.30ms with RVIZ</td><td>0.81ms no RVIZ<br>0.82ms with RVIZ</td><td>13.63ms no RVIZ<br>19.86ms with RVIZ</td>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Rotation</a></td><td>11.01ms no RVIZ<br>10.80ms with RVIZ</td><td>50.30ms no RVIZ<br>53.15ms with RVIZ</td><td>4.01ms no RVIZ<br>4.40ms with RVIZ</td><td>0.54ms no RVIZ<br>0.55ms with RVIZ</td><td>9.77ms no RVIZ<br>10.27ms with RVIZ</td>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Campus (small)</a></td><td>17.88ms no RVIZ<br>37.30ms with RVIZ</td><td>58.68ms no RVIZ<br>115.68ms with RVIZ</td><td>4.70ms no RVIZ<br>7.62ms with RVIZ</td><td>0.60ms no RVIZ<br>0.62ms with RVIZ</td><td>11.89ms no RVIZ<br>17.83ms with RVIZ</td>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">Campus (large)</a></td><td>16.20ms no RVIZ<br>28.39ms with RVIZ</td><td>60.67ms no RVIZ<br>108.08ms with RVIZ</td><td>4.76ms no RVIZ<br>7.50ms with RVIZ</td><td>0.62ms no RVIZ<br>0.63ms with RVIZ</td><td>12.48ms no RVIZ<br>17.47ms with RVIZ</td>
</tr>
<tr>
<td><a href="https://drive.google.com/drive/folders/1gJHwfdHCRdjP7vuT556pv8atqrCJPbUq?usp=sharing">2011_09_30_drive_0028</a></td><td>14.33ms no RVIZ<br>22.25ms with RVIZ</td><td>110.22ms no RVIZ<br>168.98ms with RVIZ</td><td>5.20ms no RVIZ<br>7.44ms with RVIZ</td><td>1.05ms no RVIZ<br>1.05ms with RVIZ</td><td>19.64ms no RVIZ<br>24.50ms with RVIZ</td>
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
