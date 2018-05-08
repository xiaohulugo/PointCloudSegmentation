# PointCloudSegmentation-V2

Modification: (1) use nanoflann.cpp to replace the ann library; (2) remove some cpps

Three algorithms on point cloud segmentation used in the following paper:

Pairwise Linkage for Point Cloud Segmentation, Xiaohu Lu, etc. ISPRS2016.
https://github.com/xiaohulugo/xiaohulugo.github.com/blob/master/papers/PLinkage_Point_Segmentation_ISPRS2016.pdf

*The algorithm used in the ISPRS2016 paper is ClusterGrowPLinkage.cpp

Prerequisites:
---
1. OpenCV > 2.4.x
2. OpenMP

Usage:
---
1. build the project with Cmake
2. run the code
3. see main.cpp for interfaces/demos

Performance:
---
<img src="https://github.com/xiaohulugo/images/blob/master/vehicle.jpg" width="640">
<img src="https://github.com/xiaohulugo/images/blob/master/aerial.jpg"  width="640">
<img src="https://github.com/xiaohulugo/images/blob/master/static.jpg"  width="640">

Please cite these two papers if you feel this code useful:

    @ARTICLE{Lu2016Pairwise,
    author = {Lu, Xiaohu and Yao, Jian and Tu, Jinge and Li, Kai and Li, Li and Liu, Yahui},
    title = {PAIRWISE LINKAGE FOR POINT CLOUD SEGMENTATION},
    journal = {ISPRS Annals of Photogrammetry, Remote Sensing \& Spatial Information Sciences},
    year = {2016},
    }
}
    
Feel free to correct my code, if you spotted the mistakes. You are also welcomed to Email me: fangzelu@gmail.com
