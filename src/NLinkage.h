#ifndef _NLINKAGE_H_
#define _NLINKAGE_H_
#pragma once

#include "PCAFunctions.h"
#include "opencv/cv.h"

class NLinkage 
{
public:
	NLinkage( int k, double theta, PLANE_MODE planeMode, PCA_MODE pcaMode );
	~NLinkage();

	void run( std::vector<std::vector<int> > &clusters );

	void setData( ANNpointArray pointData, int pointNum, std::vector<PCAInfo> &pcaInfos );

	void createLinkage( std::vector<PCAInfo> &pcaInfos, std::vector<int> &clusterCenterIdx, std::vector<std::vector<int> > &singleLinkage );

	void clustering( std::vector<PCAInfo> &pcaInfos, std::vector<int> &clusterCenterIdx, std::vector<std::vector<int> > &singleLinkage, std::vector<std::vector<int> > &clusters );

	void createPatch( std::vector<std::vector<int> > &clusters, std::vector<PCAInfo> &patches );

	void patchMerging( std::vector<PCAInfo> &patches, std::vector<PCAInfo> &pcaInfos, std::vector<std::vector<int> > &clusters );

	double meadian( std::vector<double> &dataset );

private:
	int k;
	double theta;
	PLANE_MODE planeMode;
	PCA_MODE pcaMode;

	int pointNum;
	ANNpointArray pointData;
	std::vector<PCAInfo> pcaInfos;
};

#endif //_NLINKAGE_H_
