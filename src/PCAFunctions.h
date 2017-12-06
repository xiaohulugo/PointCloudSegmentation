#ifndef _RDPCA_H_
#define _RDPCA_H_
#pragma once

#include "ANN.h"
#include "opencv/cv.h"

enum PLANE_MODE { PLANE, SURFACE };
enum PCA_MODE { ORIPCA, RDPCA };

struct PCAInfo
{
	double lambda0;
	cv::Matx31d normal;
	cv::Matx31d planePt;
	std::vector<int> idxIn;
	std::vector<int> idxAll;

	PCAInfo &operator =(const PCAInfo &info)
	{
		this->lambda0 = info.lambda0;
		this->normal = info.normal;
		this->idxIn = info.idxIn;
		this->idxAll = info.idxAll;

		return *this;
	}
};

class PCAFunctions 
{
public:
	PCAFunctions();
	~PCAFunctions();

	void Ori_PCA( ANNpointArray pointData, int pointNum, int k, std::vector<PCAInfo> &pcaInfos, bool outlierRemoval  );

	void HalfK_PCA( ANNpointArray pointData, int pointNum, int k, std::vector<PCAInfo> &pcaInfos, bool outlierRemoval  );

	void MCS_PCA( ANNpointArray pointData, int pointNum, int k, std::vector<PCAInfo> &pcaInfos, bool outlierRemoval  );

	void MCMD_OutlierRemoval( ANNpointArray &pointData, PCAInfo &pcaInfo );

	void PCASingle( ANNpointArray &pointData, int pointNum, PCAInfo &pcaInfo, bool outlierRemoval );

	void rdPCASingle( ANNpointArray &pointData, int pointNum, PCAInfo &pcaInfo, bool outlierRemoval );

private:
	ANNpointArray pointData;
	int pointNum;
	int k;
	std::vector<ANNidxArray> annIdxVector;
	std::vector<ANNdistArray> annDisVector;
	static double meadian( std::vector<double> dataset );
};

#endif //_RDPCA_H_
