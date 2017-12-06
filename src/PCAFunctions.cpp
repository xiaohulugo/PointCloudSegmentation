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

void PCAFunctions::Ori_PCA( ANNpointArray pointData, int pointNum, int k, std::vector<PCAInfo> &pcaInfos, bool outlierRemoval )
{
	this->pointData = pointData;
	this->pointNum = pointNum;
	this->k = k;

	// create kd-tree
	ANNkd_tree *kdTree = new ANNkd_tree( pointData, pointNum, 3 );
	for ( int i = 0; i < pointNum; ++i ) 
	{
		ANNidxArray annIdx = new ANNidx[k];
		ANNdistArray annDists = new ANNdist[k];
		kdTree->annkSearch( pointData[i], k, annIdx, annDists );

		this->annIdxVector.push_back( annIdx );
		this->annDisVector.push_back( annDists );
	}

	// PCA 
	pcaInfos.resize( pointNum );
#pragma omp parallel for
	for ( int i = 0; i < pointNum; ++i ) 
	{
		if ( i % 10000 == 0 )
		{
			cout<<i<<endl;
		}

		pcaInfos[i].idxIn.resize( k );

		// 
		cv::Matx31d h_mean( 0, 0, 0 );
		for( int j = 0; j < k; ++j )
		{
			int idx = annIdxVector[i][j];
			pcaInfos[i].idxIn[j] = idx;
			h_mean += cv::Matx31d( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
		}
		h_mean *= ( 1.0 / k );

		cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
		for( int j = 0; j < k; ++j )
		{
			int idx = annIdxVector[i][j];
			cv::Matx31d hi = cv::Matx31d( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
			h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
		}
		h_cov *=( 1.0 / k );

		// eigenvector
		cv::Matx33d h_cov_evectors;
		cv::Matx31d h_cov_evals;
		cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

		// 
		pcaInfos[i].idxAll = pcaInfos[i].idxIn;
		//pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0];
		pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0] / ( h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] );
		pcaInfos[i].normal = h_cov_evectors.row(2).t();

		// outliers removal via MCMD
		if ( ! outlierRemoval )
		{
			pcaInfos[i].idxIn = pcaInfos[i].idxAll;
		}
		else
		{
			MCMD_OutlierRemoval( pointData, pcaInfos[i] );	
		}
	}
}

void PCAFunctions::HalfK_PCA( ANNpointArray pointData, int pointNum, int k, std::vector<PCAInfo> &pcaInfos, bool outlierRemoval  )
{
	int halfK = k / 2;
	double a = 1.4826;
	double thRz = 2.5;

	this->pointData = pointData;
	this->pointNum = pointNum;
	this->k = k;

	// create kd-tree
	ANNkd_tree *kdTree = new ANNkd_tree( pointData, pointNum, 3 );
	for ( int i = 0; i < pointNum; ++i ) 
	{
		ANNidxArray annIdx = new ANNidx[k];
		ANNdistArray annDists = new ANNdist[k];
		kdTree->annkSearch( pointData[i], k, annIdx, annDists );

		this->annIdxVector.push_back( annIdx );
		this->annDisVector.push_back( annDists );
	}

	// PCA 
	pcaInfos.resize( pointNum );
#pragma omp parallel for
	for ( int i = 0; i < pointNum; ++i ) 
	{
		if ( i % 10000 == 0 )
		{
			cout<<i<<endl;
		}

		// 
		cv::Matx31d h_mean( 0, 0, 0 );
		for( int j = 0; j < halfK; ++j )
		{
			int idx = annIdxVector[i][j];
			pcaInfos[i].idxIn.push_back( idx );
			h_mean += cv::Matx31d( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
		}
		h_mean *= ( 1.0 / halfK );

		cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
		for( int j = 0; j < halfK; ++j )
		{
			int idx = annIdxVector[i][j];
			cv::Matx31d hi = cv::Matx31d( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
			h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
		}
		h_cov *=( 1.0 / halfK );

		// eigenvector
		cv::Matx33d h_cov_evectors;
		cv::Matx31d h_cov_evals;
		cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

		// 
		//pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0];
		pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0] / ( h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] );
		pcaInfos[i].normal = h_cov_evectors.row(2).t();
		for ( int j=0; j<k; ++j )
		{
			int idx = annIdxVector[i][j];
			pcaInfos[i].idxAll.push_back( idx );
		}

		// outliers removal via MCMD
		if ( ! outlierRemoval )
		{
			pcaInfos[i].idxIn = pcaInfos[i].idxAll;
		}
		else
		{
			MCMD_OutlierRemoval( pointData, pcaInfos[i] );	
		}

	}
}

void PCAFunctions::MCS_PCA( ANNpointArray pointData, int pointNum, int k, std::vector<PCAInfo> &pcaInfos, bool outlierRemoval )
{
	//
	int iterTime = log( 1 - 0.9999 ) / log( 1 - pow( 1 - 0.5, 3 ) );
	int h = k / 2;

	this->pointData = pointData;
	this->pointNum = pointNum;
	this->k = k;

	// create kd-tree
	ANNkd_tree *kdTree = new ANNkd_tree( pointData, pointNum, 3 );
	for ( int i = 0; i < pointNum; ++i ) 
	{
		ANNidxArray annIdx = new ANNidx[k];
		ANNdistArray annDists = new ANNdist[k];
		kdTree->annkSearch( pointData[i], k, annIdx, annDists );

		this->annIdxVector.push_back( annIdx );
		this->annDisVector.push_back( annDists );
	}

	pcaInfos.resize( pointNum );

	// PCA + MCS + MCMD
#pragma omp parallel for
	for ( int curIdx = 0; curIdx < pointNum; ++curIdx ) 
	{
		if ( curIdx % 10000 == 0 )
		{
			cout<<curIdx<<endl;
		}

		// MCS
		std::vector<std::vector<ANNidx> > h_subset_idx_vec;
		for ( int iter = 0; iter < iterTime; ++iter ) 
		{
			ANNidx h0_idx0 = rand() % k;
			ANNidx h0_idx1 = rand() % k;
			ANNidx h0_idx2 = rand() % k;

			ANNpoint ann_h0_0 = pointData[annIdxVector[curIdx][h0_idx0]];
			ANNpoint ann_h0_1 = pointData[annIdxVector[curIdx][h0_idx1]];
			ANNpoint ann_h0_2 = pointData[annIdxVector[curIdx][h0_idx2]];

			cv::Matx31d h0_0(ann_h0_0[0], ann_h0_0[1], ann_h0_0[2]);
			cv::Matx31d h0_1(ann_h0_1[0], ann_h0_1[1], ann_h0_1[2]);
			cv::Matx31d h0_2(ann_h0_2[0], ann_h0_2[1], ann_h0_2[2]);
			cv::Matx31d h0_mean = ( h0_0 + h0_1 + h0_2 ) * ( 1.0 / 3.0 );

			// PCA
			cv::Matx33d h0_cov = ( ( h0_0 - h0_mean ) * ( h0_0 - h0_mean ).t()
				+ ( h0_1 - h0_mean ) * ( h0_1 - h0_mean ).t() 
				+ ( h0_2 - h0_mean ) * ( h0_2 - h0_mean ).t() ) * ( 1.0 / 3.0 );

			cv::Matx33d h0_cov_evectors;
			cv::Matx31d h0_cov_evals;
			cv::eigen( h0_cov, h0_cov_evals, h0_cov_evectors );

			// OD
			std::vector<std::pair<int, double> > ODs( k );
			for ( int i = 0; i < k; ++i ) 
			{
				cv::Matx31d ptMat( pointData[annIdxVector[curIdx][i]][0], pointData[annIdxVector[curIdx][i]][1], pointData[annIdxVector[curIdx][i]][2] );
				cv::Matx<double, 1, 1> ODMat = ( ptMat - h0_mean ).t() * h0_cov_evectors.row(2).t();
				double OD = fabs( ODMat.val[0] );
				ODs[i].first = annIdxVector[curIdx][i];
				ODs[i].second = OD;
			}
			std::sort( ODs.begin(), ODs.end(), [](const std::pair<int,double>& lhs, const std::pair<int,double>& rhs) { return lhs.second < rhs.second; } );

			// h-subset
			std::vector<ANNidx> h_subset_idx;
			for( int i = 0; i < h; i++ )
			{
				h_subset_idx.push_back( ODs[i].first );
			}
			h_subset_idx_vec.push_back( h_subset_idx );
		}

		// calculate the PCA hypotheses
		std::vector<PCAInfo> S_PCA;
		S_PCA.resize( h_subset_idx_vec.size() );
		for ( int i = 0; i < h_subset_idx_vec.size(); ++i ) 
		{
			cv::Matx31d h_mean( 0, 0, 0 );
			for( int j = 0; j < h; ++j )
			{
				int index = h_subset_idx_vec[i][j];
				h_mean += cv::Matx31d( pointData[index][0], pointData[index][1], pointData[index][2] );
			}
			h_mean *= ( 1.0 / h );

			cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
			for( int j = 0; j < h; ++j )
			{
				int index = h_subset_idx_vec[i][j];
				cv::Matx31d hi = cv::Matx31d( pointData[index][0], pointData[index][1], pointData[index][2] );
				h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
			}
			h_cov *=( 1.0 / h );

			//
			cv::Matx33d h_cov_evectors;
			cv::Matx31d h_cov_evals;
			cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

			S_PCA[i].idxIn = h_subset_idx_vec[i];
			//S_PCA[i].lambda0 = h_cov_evals.row(2).val[0];
			S_PCA[i].lambda0 = h_cov_evals.row(2).val[0] / ( h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] );
			S_PCA[i].normal = h_cov_evectors.row(2).t();
		}
		std::sort( S_PCA.begin(), S_PCA.end(), [](const PCAInfo& lhs, const PCAInfo& rhs) { return lhs.lambda0 < rhs.lambda0; } );

		pcaInfos[curIdx] = S_PCA[0];
		pcaInfos[curIdx].idxAll.resize( k );
		for ( int i = 0; i < k; ++i ) 
		{
			pcaInfos[curIdx].idxAll[i] = annIdxVector[curIdx][i];
		}

		// outliers removal via MCMD
		if ( ! outlierRemoval )
		{
			pcaInfos[curIdx].idxIn = pcaInfos[curIdx].idxAll;
		}
		else
		{
			MCMD_OutlierRemoval( pointData, pcaInfos[curIdx] );	
		}
	}
}

void PCAFunctions::MCMD_OutlierRemoval( ANNpointArray &pointData, PCAInfo &pcaInfo )
{
	double a = 1.4826;
	double thRz = 2.5;
	int num = pcaInfo.idxAll.size();

	// ODs
	cv::Matx31d h_mean( 0, 0, 0 );
	for( int j = 0; j < pcaInfo.idxIn.size(); ++j )
	{
		int idx = pcaInfo.idxIn[j];
		h_mean += cv::Matx31d( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
	}
	h_mean *= ( 1.0 / pcaInfo.idxIn.size() );

	std::vector<double> ODs( num );
	for( int j = 0; j < num; ++j )
	{
		int idx = pcaInfo.idxAll[j];
		cv::Matx31d pt( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
		cv::Matx<double, 1, 1> OD_mat = ( pt - h_mean ).t() * pcaInfo.normal;
		double OD = fabs( OD_mat.val[0] );
		ODs[j] = OD;
	}

	// calculate the Rz-score for all points using ODs
	std::vector<double> sorted_ODs( ODs.begin(), ODs.end() );
	double median_OD = meadian( sorted_ODs );
	std::vector<double>().swap( sorted_ODs );

	std::vector<double> abs_diff_ODs( num );
	for( int j = 0; j < num; ++j )
	{
		abs_diff_ODs[j] = fabs( ODs[j] - median_OD );
	}
	double MAD_OD = a * meadian( abs_diff_ODs );
	std::vector<double>().swap( abs_diff_ODs );

	// get inlier 
	std::vector<int> idxInlier;
	for( int j = 0; j < num; ++j )
	{
		double Rzi = fabs( ODs[j] - median_OD ) / MAD_OD;
		if ( Rzi < thRz ) 
		{
			int idx = pcaInfo.idxAll[j];
			idxInlier.push_back( idx );
		}
	}

	// 
	pcaInfo.idxIn = idxInlier;
}

void PCAFunctions::rdPCASingle( ANNpointArray &pointData, int pointNum, PCAInfo &pcaInfo, bool outlierRemoval )
{
	int i, j;
	int h0 = 3;
	int k = pointNum;
	int h = k / 2;
	int iterTime = log( 1 - 0.9999 ) / log( 1 - pow( 1 - 0.5, h0 ) );
	double a = 1.4826;
	double thRz = 2.5;

	std::vector<std::vector<ANNidx> > h_subset_idx_vec;
	for ( int iter = 0; iter < iterTime; ++iter ) 
	{
		int h0_idx0 = rand() % k;
		int h0_idx1 = rand() % k;
		int h0_idx2 = rand() % k;

		cv::Matx31d h0_0( pointData[h0_idx0][0], pointData[h0_idx0][1], pointData[h0_idx0][2] );
		cv::Matx31d h0_1( pointData[h0_idx1][0], pointData[h0_idx1][1], pointData[h0_idx1][2] );
		cv::Matx31d h0_2( pointData[h0_idx2][0], pointData[h0_idx2][1], pointData[h0_idx2][2] );
		cv::Matx31d h0_mean = ( h0_0 + h0_1 + h0_2 ) * ( 1.0 / 3.0 );

		// PCA
		cv::Matx33d h0_cov = ( ( h0_0 - h0_mean ) * ( h0_0 - h0_mean ).t()
			+ ( h0_1 - h0_mean ) * ( h0_1 - h0_mean ).t() 
			+ ( h0_2 - h0_mean ) * ( h0_2 - h0_mean ).t() ) * ( 1.0 / 3.0 );

		cv::Matx33d h0_cov_evectors;
		cv::Matx31d h0_cov_evals;
		cv::eigen( h0_cov, h0_cov_evals, h0_cov_evectors );

		// OD
		std::vector<std::pair<int, double> > ODs( k );
		for ( i = 0; i < k; ++i ) 
		{
			cv::Matx31d ptMat( pointData[i][0], pointData[i][1], pointData[i][2] );
			cv::Matx<double, 1, 1> ODMat = ( ptMat - h0_mean ).t() * h0_cov_evectors.row(2).t();
			double OD = fabs( ODMat.val[0] );
			ODs[i].first = i;
			ODs[i].second = OD;
		}
		std::sort( ODs.begin(), ODs.end(), [](const std::pair<int,double>& lhs, const std::pair<int,double>& rhs) { return lhs.second < rhs.second; } );


		// h-subset
		std::vector<ANNidx> h_subset_idx;
		for( i = 0; i < h; i++ )
		{
			h_subset_idx.push_back( ODs[i].first );
		}
		h_subset_idx_vec.push_back( h_subset_idx );
	}

	// calculate the PCA hypotheses
	std::vector<PCAInfo> S_PCA;
	S_PCA.resize( h_subset_idx_vec.size() );
	for ( i = 0; i < h_subset_idx_vec.size(); ++i ) 
	{
		cv::Matx31d h_mean( 0, 0, 0 );
		for( j = 0; j < h; ++j )
		{
			int index = h_subset_idx_vec[i][j];
			h_mean += cv::Matx31d( pointData[index][0], pointData[index][1], pointData[index][2] );
		}
		h_mean *= ( 1.0 / double( h ) );

		cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
		for( j = 0; j < h; ++j )
		{
			int index = h_subset_idx_vec[i][j];
			cv::Matx31d hi = cv::Matx31d( pointData[index][0], pointData[index][1], pointData[index][2] );
			h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
		}
		h_cov *= ( 1.0 / double( h ) );

		//
		cv::Matx33d h_cov_evectors;
		cv::Matx31d h_cov_evals;
		cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

		PCAInfo pcaEles;
		//pcaEles.lambda0 = h_cov_evals.row(2).val[0];
		pcaEles.lambda0 = h_cov_evals.row(2).val[0] / ( h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] );
		pcaEles.normal = h_cov_evectors.row(2).t();
		pcaEles.idxIn = h_subset_idx_vec[i];
		S_PCA[i] = pcaEles;
	}

	// find the best PCA
	std::sort( S_PCA.begin(), S_PCA.end(), [](const PCAInfo& lhs, const PCAInfo& rhs) { return lhs.lambda0 < rhs.lambda0; } );
	pcaInfo.lambda0 = S_PCA[0].lambda0;

	pcaInfo.normal = S_PCA[0].normal;
	double N = sqrt( pcaInfo.normal.val[0] * pcaInfo.normal.val[0] + pcaInfo.normal.val[1] * pcaInfo.normal.val[1] + pcaInfo.normal.val[2] * pcaInfo.normal.val[2] );
	pcaInfo.normal *= 1.0 / N;

	pcaInfo.idxAll.resize( k );
	for ( i = 0; i < k; ++i ) 
	{
		pcaInfo.idxAll[i] = i;
	}

	// outliers removal via MCMD
	if ( ! outlierRemoval )
	{
		pcaInfo.idxIn = pcaInfo.idxAll;
	}
	else
	{
		MCMD_OutlierRemoval( pointData, pcaInfo );	
	}
}

void PCAFunctions::PCASingle( ANNpointArray &pointData, int pointNum, PCAInfo &pcaInfo, bool outlierRemoval )
{
	int i, j;
	int k = pointNum;
	double a = 1.4826;
	double thRz = 2.5;

	// 
	pcaInfo.idxIn.resize( k );
	cv::Matx31d h_mean( 0, 0, 0 );
	for( i = 0; i < k; ++i )
	{
		pcaInfo.idxIn[i] = i;
		h_mean += cv::Matx31d( pointData[i][0], pointData[i][1], pointData[i][2] );
	}
	h_mean *= ( 1.0 / k );

	cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	for( i = 0; i < k; ++i )
	{
		cv::Matx31d hi = cv::Matx31d( pointData[i][0], pointData[i][1], pointData[i][2] );
		h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
	}
	h_cov *=( 1.0 / k );

	// eigenvector
	cv::Matx33d h_cov_evectors;
	cv::Matx31d h_cov_evals;
	cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

	// 
	pcaInfo.idxAll = pcaInfo.idxIn;
	//pcaInfo.lambda0 = h_cov_evals.row(2).val[0];
	pcaInfo.lambda0 = h_cov_evals.row(2).val[0] / ( h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] );
	pcaInfo.normal = h_cov_evectors.row(2).t();

	// outliers removal via MCMD
	if ( ! outlierRemoval )
	{
		pcaInfo.idxIn = pcaInfo.idxAll;
	}
	else
	{
		MCMD_OutlierRemoval( pointData, pcaInfo );	
	}
}

double PCAFunctions::meadian( std::vector<double> dataset )
{
	std::sort( dataset.begin(), dataset.end(), []( const double& lhs, const double& rhs ){ return lhs < rhs; } );
	if(dataset.size()%2 == 0)
	{
		return dataset[dataset.size()/2];
	}
	else
	{
		return (dataset[dataset.size()/2] + dataset[dataset.size()/2 + 1])/2.0;
	}
}