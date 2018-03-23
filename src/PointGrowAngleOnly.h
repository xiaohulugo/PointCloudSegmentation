//  Title   = {Segmentation of point clouds using smoothness constraint},
//  Author  = {Rabbani, Tahir and van den Heuvel, Frank and Vosselmann, G},
//  Journal = {\textit{International Archives of Photogrammetry, Remote Sensing and Spatial Information Sciences}},
//  Year    = {2006},

#ifndef _SMOOTH_CONSTRAINT_H_
#define _SMOOTH_CONSTRAINT_H_
#pragma once

#include "PCAFunctions.h"
#include "opencv/cv.h"

class SmoothConstraint 
{
public:
	SmoothConstraint( double theta, double percent );
	~SmoothConstraint();

	void run( std::vector<std::vector<int> > &clusters );

	void setData( ANNpointArray pointData, int pointNum, std::vector<PCAInfo> &pcaInfos );

private:
	 double theta;
	 double percent;

	 int pointNum;
	 ANNpointArray pointData;
	 std::vector<PCAInfo> pcaInfos;
};

#endif //_SMOOTH_CONSTRAINT_H_
