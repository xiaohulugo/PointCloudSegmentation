#include "PCAFunctions.h"
#include <stdio.h>
#include <omp.h>

using namespace std;

PCAFunctions::PCAFunctions()
{
}

PCAFunctions::~PCAFunctions()
{

}
void PCAFunctions::Ori_PCA(PointCloud<double> &cloud, int k, std::vector<PCAInfo> &pcaInfos, double &scale, double &magnitd)
{
	double MINVALUE = 1e-7;
	int pointNum = cloud.pts.size();

	// 1. build kd-tree
	typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud<double> >, PointCloud<double>, 3/*dim*/ > my_kd_tree_t;
	my_kd_tree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	// 2. knn search
	size_t *out_ks = new size_t[pointNum];
	size_t **out_indices = new size_t *[pointNum];
#pragma omp parallel for
	for (int i = 0; i<pointNum; ++i)
	{
		double *query_pt = new double[3];
		query_pt[0] = cloud.pts[i].x;  query_pt[1] = cloud.pts[i].y;  query_pt[2] = cloud.pts[i].z;
		double *dis_temp = new double[k];
		out_indices[i] = new size_t[k];

		nanoflann::KNNResultSet<double> resultSet(k);
		resultSet.init(out_indices[i], dis_temp);
		index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
		out_ks[i] = resultSet.size();

		delete query_pt;
		delete dis_temp;
	}
	index.freeIndex(index);

	// 3. PCA normal estimation
	scale = 0.0;
	pcaInfos.resize(pointNum);
#pragma omp parallel for
	for (int i = 0; i < pointNum; ++i)
	{
		// 
		int ki = out_ks[i];

		double h_mean_x = 0.0, h_mean_y = 0.0, h_mean_z = 0.0;
		for (int j = 0; j < ki; ++j)
		{
			int idx = out_indices[i][j];
			h_mean_x += cloud.pts[idx].x;
			h_mean_y += cloud.pts[idx].y;
			h_mean_z += cloud.pts[idx].z;
		}
		h_mean_x *= 1.0 / ki;  h_mean_y *= 1.0 / ki; h_mean_z *= 1.0 / ki;

		double h_cov_1 = 0.0, h_cov_2 = 0.0, h_cov_3 = 0.0;
		double h_cov_5 = 0.0, h_cov_6 = 0.0;
		double h_cov_9 = 0.0;
		double dx = 0.0, dy = 0.0, dz = 0.0;
		for (int j = 0; j < k; ++j)
		{
			int idx = out_indices[i][j];
			dx = cloud.pts[idx].x - h_mean_x;
			dy = cloud.pts[idx].y - h_mean_y;
			dz = cloud.pts[idx].z - h_mean_z;

			h_cov_1 += dx * dx; h_cov_2 += dx * dy; h_cov_3 += dx * dz;
			h_cov_5 += dy * dy; h_cov_6 += dy * dz;
			h_cov_9 += dz * dz;
		}
		cv::Matx33d h_cov(
			h_cov_1, h_cov_2, h_cov_3,
			h_cov_2, h_cov_5, h_cov_6,
			h_cov_3, h_cov_6, h_cov_9);
		h_cov *= 1.0 / ki;

		// eigenvector
		cv::Matx33d h_cov_evectors;
		cv::Matx31d h_cov_evals;
		cv::eigen(h_cov, h_cov_evals, h_cov_evectors);

		// 
		pcaInfos[i].idxAll.resize(ki);
		for (int j = 0; j<ki; ++j)
		{
			int idx = out_indices[i][j];
			pcaInfos[i].idxAll[j] = idx;
		}

		int idx = out_indices[i][3];
		dx = cloud.pts[idx].x - cloud.pts[i].x;
		dy = cloud.pts[idx].y - cloud.pts[i].y;
		dz = cloud.pts[idx].z - cloud.pts[i].z;
		double scaleTemp = sqrt(dx*dx + dy * dy + dz * dz);
		pcaInfos[i].scale = scaleTemp;
		scale += scaleTemp;

		//pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0];
		double t = h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] + (rand() % 10 + 1) * MINVALUE;
		pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0] / t;
		pcaInfos[i].normal = h_cov_evectors.row(2).t();

		// outliers removal via MCMD
		pcaInfos[i].idxIn = pcaInfos[i].idxAll;

		delete out_indices[i];
	}
	delete[]out_indices;
	delete out_ks;

	scale /= pointNum;
	magnitd = sqrt(cloud.pts[0].x*cloud.pts[0].x + cloud.pts[0].y*cloud.pts[0].y + cloud.pts[0].z*cloud.pts[0].z);
}

void PCAFunctions::RDPCA(PointCloud<double>& cloud, int k, std::vector<PCAInfo>& pcaInfos, double & scale, double & magnitd)
{
}

void PCAFunctions::PCASingle(std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo)
{
	int i, j;
	int k = pointData.size();
	double a = 1.4826;
	double thRz = 2.5;

	// 
	pcaInfo.idxIn.resize(k);
	cv::Matx31d h_mean(0, 0, 0);
	for (i = 0; i < k; ++i)
	{
		pcaInfo.idxIn[i] = i;
		h_mean += cv::Matx31d(pointData[i][0], pointData[i][1], pointData[i][2]);
	}
	h_mean *= (1.0 / k);

	cv::Matx33d h_cov(0, 0, 0, 0, 0, 0, 0, 0, 0);
	for (i = 0; i < k; ++i)
	{
		cv::Matx31d hi = cv::Matx31d(pointData[i][0], pointData[i][1], pointData[i][2]);
		h_cov += (hi - h_mean) * (hi - h_mean).t();
	}
	h_cov *= (1.0 / k);

	// eigenvector
	cv::Matx33d h_cov_evectors;
	cv::Matx31d h_cov_evals;
	cv::eigen(h_cov, h_cov_evals, h_cov_evectors);

	// 
	pcaInfo.idxAll = pcaInfo.idxIn;
	//pcaInfo.lambda0 = h_cov_evals.row(2).val[0];
	pcaInfo.lambda0 = h_cov_evals.row(2).val[0] / (h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0]);
	pcaInfo.normal = h_cov_evectors.row(2).t();
	pcaInfo.planePt = h_mean;

	// outliers removal via MCMD
	MCMD_OutlierRemoval(pointData, pcaInfo);
}

void PCAFunctions::RDPCASingle(std::vector<std::vector<double>>& pointData, PCAInfo & pcaInfo)
{
}

void PCAFunctions::MCMD_OutlierRemoval(std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo)
{
	double a = 1.4826;
	double thRz = 2.5;
	int num = pcaInfo.idxAll.size();

	// ODs
	cv::Matx31d h_mean(0, 0, 0);
	for (int j = 0; j < pcaInfo.idxIn.size(); ++j)
	{
		int idx = pcaInfo.idxIn[j];
		h_mean += cv::Matx31d(pointData[idx][0], pointData[idx][1], pointData[idx][2]);
	}
	h_mean *= (1.0 / pcaInfo.idxIn.size());

	std::vector<double> ODs(num);
	for (int j = 0; j < num; ++j)
	{
		int idx = pcaInfo.idxAll[j];
		cv::Matx31d pt(pointData[idx][0], pointData[idx][1], pointData[idx][2]);
		cv::Matx<double, 1, 1> OD_mat = (pt - h_mean).t() * pcaInfo.normal;
		double OD = fabs(OD_mat.val[0]);
		ODs[j] = OD;
	}

	// calculate the Rz-score for all points using ODs
	std::vector<double> sorted_ODs(ODs.begin(), ODs.end());
	double median_OD = meadian(sorted_ODs);
	std::vector<double>().swap(sorted_ODs);

	std::vector<double> abs_diff_ODs(num);
	for (int j = 0; j < num; ++j)
	{
		abs_diff_ODs[j] = fabs(ODs[j] - median_OD);
	}
	double MAD_OD = a * meadian(abs_diff_ODs) + 1e-6;
	std::vector<double>().swap(abs_diff_ODs);

	// get inlier 
	std::vector<int> idxInlier;
	for (int j = 0; j < num; ++j)
	{
		double Rzi = fabs(ODs[j] - median_OD) / MAD_OD;
		if (Rzi < thRz)
		{
			int idx = pcaInfo.idxAll[j];
			idxInlier.push_back(idx);
		}
	}

	// 
	pcaInfo.idxIn = idxInlier;
}

double PCAFunctions::meadian(std::vector<double> dataset)
{
	std::sort(dataset.begin(), dataset.end(), [](const double& lhs, const double& rhs) { return lhs < rhs; });
	if (dataset.size() % 2 == 0)
	{
		return dataset[dataset.size() / 2];
	}
	else
	{
		return (dataset[dataset.size() / 2] + dataset[dataset.size() / 2 + 1]) / 2.0;
	}
}