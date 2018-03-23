#ifndef _POINT_CLOUD_SEGMENTATION_H_
#define _POINT_CLOUD_SEGMENTATION_H_
#pragma once

#include <stdio.h>
#include "PCAFunctions.h"
#include "nanoflann.hpp"
#include "opencv/cv.h"

enum ALGORITHM { PLINKAGE, REGIONGROW, SMOOTHCONSTRAINT, RBNNN };

class PointCloudSegmentation 
{
public:
	PointCloudSegmentation();
	~PointCloudSegmentation();

	void setData(PointCloud<double> &data, std::vector<PCAInfo> &pcaInfos);

	void run( ALGORITHM algorithm, std::vector<std::vector<int> > &clusters );

	void writeOutClusters( std::string filePath, std::vector<std::vector<int> > &clusters );

	void writeOutClusters2( std::string filePath, std::vector<std::vector<int> > &clusters );

private:
	int pointNum;
	PointCloud<double> pointData;
	std::vector<PCAInfo> pcaInfos;
};

#endif //_POINT_CLOUD_SEGMENTATION_H_
