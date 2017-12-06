#ifndef _RBNN_H_
#define _RBNN_H_
#pragma once

#include "PCAFunctions.h"
#include "opencv/cv.h"

class RBNN 
{
public:
	RBNN( double voxelSize, double r, int nMin );
	~RBNN();

	void run( std::vector<std::vector<int> > &clusters );

	void setData( ANNpointArray pointData, int pointNum );

	void findNeighborInRadius( std::vector<std::vector<std::vector<std::vector<int> > > > &voxels, int idxPt, cv::Point3i &idxVoxel, std::vector<int> &idxNeib );
private:
	double voxelSize; 
	double r;
	int nMin;
	int pointNum;
	ANNpointArray pointData;
};

#endif //_RBNN_H_
