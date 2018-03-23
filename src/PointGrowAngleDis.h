#ifndef _REGION_GROW_H_
#define _REGION_GROW_H_
#pragma once

#include "PCAFunctions.h"
#include "opencv/cv.h"

class RegionGrow 
{
public:
	RegionGrow( double theta, int Rmin );
	~RegionGrow();

	void run( std::vector<std::vector<int> > &clusters );

	void setData( ANNpointArray pointData, int pointNum, std::vector<PCAInfo> &pcaInfos );

	double meadian( std::vector<double> &dataset );

private:
	double theta; 
	int Rmin;

	int pointNum;
	ANNpointArray pointData;
	std::vector<PCAInfo> pcaInfos;
};

#endif //_NLINKAGE_H_
