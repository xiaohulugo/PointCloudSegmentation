#ifndef _PLINKAGE_H_
#define _PLINKAGE_H_
#pragma once

#include "PCAFunctions.h"
#include "opencv/cv.h"

class PLinkage 
{
public:
	PLinkage( int k, double theta, PLANE_MODE planeMode, PCA_MODE pcaMode );
	~PLinkage();

	void run( std::vector<std::vector<int> > &clusters );

	void setData(PointCloud<double> &data, std::vector<PCAInfo> &pcaInfos );

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
	PointCloud<double> pointData;
	std::vector<PCAInfo> pcaInfos;
};

#endif //_PLINKAGE_H_
