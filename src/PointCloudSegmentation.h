#ifndef _POINT_CLOUD_SEGMENTATION_H_
#define _POINT_CLOUD_SEGMENTATION_H_
#pragma once

#include <stdio.h>
#include "ANN.h"
#include "opencv/cv.h"

enum ALGORITHM { PLINKAGE, REGIONGROW, SMOOTHCONSTRAINT, RBNNN };

class PointCloudSegmentation 
{
public:
	PointCloudSegmentation();
	~PointCloudSegmentation();

	void run( std::string filepath, ALGORITHM algorithm, std::vector<std::vector<int> > &clusters );

	bool setDataFromFile( std::string filepath );

	void setDataFromANN( ANNpointArray pointData, int pointNum );

	void writeOutClusters( std::string filePath, std::vector<std::vector<int> > &clusters );

	void writeOutClusters2( std::string filePath, std::vector<std::vector<int> > &clusters );

private:
	int pointNum;
	ANNpointArray pointData;
};

#endif //_POINT_CLOUD_SEGMENTATION_H_
