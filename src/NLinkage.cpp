#include "NLinkage.h"
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include "highgui.h"

using namespace std;

NLinkage::NLinkage( int k, double theta, PLANE_MODE planeMode, PCA_MODE pcaMode )
{
	this->k = k;
	this->theta = theta;
	this->planeMode = planeMode;
	this->pcaMode = pcaMode;
}

NLinkage::~NLinkage()
{
}

void NLinkage::setData( ANNpointArray pointData, int pointNum, std::vector<PCAInfo> &pcaInfos )
{
	this->pointData = pointData;
	this->pointNum = pointNum;
	this->pcaInfos = pcaInfos;
}

void NLinkage::run( std::vector<std::vector<int> > &clusters )
{
	// create linkage
	std::vector<int> clusterCenterIdx;
	std::vector<std::vector<int> > singleLinkage;
	createLinkage( pcaInfos, clusterCenterIdx, singleLinkage );

	// data clustering
	std::vector<std::vector<int> > clustersInit;
	clustering( pcaInfos, clusterCenterIdx, singleLinkage, clustersInit );

	// create patches via RDPCA
	std::vector<PCAInfo> patchesInit;
	createPatch( clustersInit, patchesInit );

	// patch merging
	patchMerging( patchesInit, pcaInfos, clusters );
}

void NLinkage::createLinkage( std::vector<PCAInfo> &pcaInfos, std::vector<int> &clusterCenterIdx, std::vector<std::vector<int> > &singleLinkage )
{
	cout<<"create linkage"<<endl;

	// get the threshold of lambda0
	std::vector<double> lambdas( pcaInfos.size() );
	double lambdaAvg = 0.0;
	for ( int i=0; i<pcaInfos.size(); ++i )
	{
		lambdas[i] = pcaInfos[i].lambda0;
		lambdaAvg += lambdas[i];
	}
	lambdaAvg /= double( pcaInfos.size() );

	double lambdaVar = 0.0;
	for ( int i=0;  i<pcaInfos.size(); ++i )
	{
		lambdaVar += ( lambdas[i] - lambdaAvg ) * ( lambdas[i] - lambdaAvg );
	}

	double lambdaSTD = lambdaAvg + sqrt( lambdaVar / pcaInfos.size() );
	//double lambdaSTD =  lambdaAvg;
	cout<<"lambdaSTD: "<<lambdaSTD<<endl;

	// create linkages
	std::vector<std::vector<int> > linkage( this->pointNum );
	std::vector<int> clusterCenter;
	for ( int i=0; i<this->pointNum; ++i )
	{
		if ( i % 10000 == 0 )
		{
			cout<<i<<endl;
		}

		if ( i >= 100 )
		{
			int aa=0;
		}
		int idx = i;
		double lambda = pcaInfos[idx].lambda0;
		double x = pointData[idx][0];
		double y = pointData[idx][1];
		double z = pointData[idx][2];
		cv::Matx31d normal = pcaInfos[idx].normal;
		double N = sqrt( normal.val[0] * normal.val[0] + normal.val[1] * normal.val[1] + normal.val[2] * normal.val[2] );
		normal *= double(1.0) / N;

		// find the closest neighbor point which is more likely to make a plane
		std::vector<double> devAngles( pcaInfos[idx].idxIn.size(), 100000.0 );
#pragma omp parallel for
		for ( int j=0; j<pcaInfos[idx].idxIn.size(); ++j )
		{
			int idxCur = pcaInfos[idx].idxIn[j];
			double lambdaCur = pcaInfos[idxCur].lambda0;
			if ( lambdaCur >= lambda )
			{
				continue;
			}

			bool isIn = false;
			for ( int m=0; m<pcaInfos[idxCur].idxIn.size(); ++m )
			{
				if ( pcaInfos[idxCur].idxIn[m] == idx )
				{
					isIn = true;
					break;
				}
			}
			if ( !isIn )
			{
				continue;
			}

			//
			double xCur = pointData[idxCur][0];
			double yCur = pointData[idxCur][1];
			double zCur = pointData[idxCur][2];
			cv::Matx31d normalCur = pcaInfos[idxCur].normal;
			double NCur = sqrt( normalCur.val[0] * normalCur.val[0] + normalCur.val[1] * normalCur.val[1] + normalCur.val[2] * normalCur.val[2] );
			normalCur *= double(1.0) / NCur;

			// plane distance
			cv::Matx31d offset( xCur - x, yCur - y, zCur - z );
			cv::Matx<double, 1, 1> ODMat1 = normal.t() * offset;
			cv::Matx<double, 1, 1> ODMat2 = normalCur.t() * offset;

			double OD1 = fabs( ODMat1.val[0] );
			double OD2 = fabs( ODMat2.val[0] );
			double devDis = ( OD1 + OD2 ) / 2.0;

			// normal deviation
			double devAngle = acos( normal.val[0] * normalCur.val[0] + normal.val[1] * normalCur.val[1] + normal.val[2] * normalCur.val[2] );
			devAngle = min( devAngle, CV_PI - devAngle );
			devAngles[j] = devAngle;
		}

		// find the neighbor point with closest normal direction
		int idxBest = -1;
		double devMin = 1000.0;
		for ( int j=0; j<devAngles.size(); ++j )
		{
			if ( devAngles[j] < devMin )
			{
				idxBest = pcaInfos[idx].idxIn[j];
				devMin = devAngles[j];
			}
		}

		if ( idxBest >= 0 )             // find a single linkage
		{
			linkage[idxBest].push_back( idx );
		}
		else if ( lambda < lambdaSTD )
		{
			clusterCenter.push_back( idx );
		}
	}

	//
	singleLinkage = linkage;
	clusterCenterIdx = clusterCenter;
}

void NLinkage::clustering( std::vector<PCAInfo> &pcaInfos, std::vector<int> &clusterCenterIdx, std::vector<std::vector<int> > &singleLinkage, std::vector<std::vector<int> > &clusters )
{
	cout<<"clustering"<<endl;
	int i, j;
	double thAngleDev = 10.0 / 180.0 * CV_PI;
	//
	int count1 = 0;
	while ( count1 < clusterCenterIdx.size() )
	{
		int idxCenter = clusterCenterIdx[count1];
		cv::Matx31d normalCenter = pcaInfos[idxCenter].normal;

		std::vector<int> seedIdx;
		seedIdx.push_back( idxCenter );

		std::vector<int> cluster;
		cluster.push_back( idxCenter );

		int count2 = 0;
		while ( count2 < seedIdx.size() )
		{
			int idxSeed = seedIdx[count2];
			cv::Matx31d normalSeed = pcaInfos[idxCenter].normal;

			for ( j=0; j<singleLinkage[idxSeed].size(); ++j )
			{
				int idxPt = singleLinkage[idxSeed][j];
				cv::Matx31d normalPt = pcaInfos[idxPt].normal;

				double devAngle = 0.0;
				if ( this->planeMode == PLANE )
				{
					devAngle = acos( normalCenter.val[0] * normalPt.val[0] + normalCenter.val[1] * normalPt.val[1] + normalCenter.val[2] * normalPt.val[2] );
				}
				else
				{
					devAngle = acos( normalSeed.val[0] * normalPt.val[0] + normalSeed.val[1] * normalPt.val[1] + normalSeed.val[2] * normalPt.val[2] );
				}

				if ( devAngle < thAngleDev )
				{
					cluster.push_back( idxPt );
					seedIdx.push_back( idxPt );
				}
				else
				{
					clusterCenterIdx.push_back( idxPt );
				}

			}
			count2 ++;
		}

		if ( cluster.size() >= 10 )
		{
			clusters.push_back( cluster );
		}

		count1 ++;
	}

	/*
	//
	int numCenter = clusterCenterIdx.size();

	for ( i=0; i<numCenter; ++i )
	{
		std::vector<int> seedIdx;
		seedIdx.push_back( clusterCenterIdx[i] );

		std::vector<int> cluster;
		cluster.push_back( clusterCenterIdx[i] );

		int count = 0;
		while ( count < seedIdx.size() )
		{
			int idxCur = seedIdx[count];
			for ( j=0; j<singleLinkage[idxCur].size(); ++j )
			{
				cluster.push_back( singleLinkage[idxCur][j] );
				seedIdx.push_back( singleLinkage[idxCur][j] );
			}

			count ++;
		}

		if ( cluster.size() >= 10 )
		{
			clusters.push_back( cluster );
		}
	}
	*/

}

void NLinkage::createPatch( std::vector<std::vector<int> > &clusters, std::vector<PCAInfo> &patches )
{
	cout<<" clusters number: "<<clusters.size()<<endl;
	cout<<"create patch RDPCA"<<endl;

	int numCluster = clusters.size();
	patches.resize( numCluster );

	// plane outliers removal via RDPCA
#pragma omp parallel for
	for ( int i=0; i<numCluster; ++i )
	{
		int pointNumCur = clusters[i].size();
		ANNpointArray pointDataCur = annAllocPts( pointNumCur, 3 );
		for ( int j=0; j<pointNumCur; ++j )
		{
			pointDataCur[j][0] = this->pointData[clusters[i][j]][0];
			pointDataCur[j][1] = this->pointData[clusters[i][j]][1];
			pointDataCur[j][2] = this->pointData[clusters[i][j]][2];
		}
		
		PCAFunctions pcaer;
		if ( this->pcaMode == ORIPCA )
		{
			pcaer.PCASingle( pointDataCur, pointNumCur, patches[i], true );
		}
		else
		{
			pcaer.rdPCASingle( pointDataCur, pointNumCur, patches[i], true );
		}

		//PCAFunctions::rdPCASingle( pointDataCur, patches[i] );

		patches[i].idxAll = clusters[i];
		patches[i].planePt = cv::Matx31d( 0.0, 0.0, 0.0 );
		for ( int j=0; j<patches[i].idxIn.size(); ++j )
		{
			int idx = patches[i].idxIn[j];
			patches[i].idxIn[j] = clusters[i][idx];

			patches[i].planePt.val[0] += this->pointData[clusters[i][idx]][0];
			patches[i].planePt.val[1] += this->pointData[clusters[i][idx]][1];
			patches[i].planePt.val[2] += this->pointData[clusters[i][idx]][2];
		}
		patches[i].planePt *= ( 1.0 / double( patches[i].idxIn.size() ) );
	}
}

void NLinkage::patchMerging( std::vector<PCAInfo> &patches, std::vector<PCAInfo> &pcaInfos, std::vector<std::vector<int> > &clusters )
{
	cout<<"patch merging"<<endl;

	double a = 1.4826;
	double minAngle = 10.0 / 180.0 * CV_PI;
	double maxAngle = this->theta;

	int numPatch = patches.size();
	// sort the patches by point number
	std::sort( patches.begin(), patches.end(), [](const PCAInfo& lhs, const PCAInfo& rhs) { return lhs.idxIn.size() > rhs.idxIn.size(); } );

	// calculate the angle deviation threshold of each patch
	std::vector<double> surfaceVariance( patches.size() );
	for ( int i=0; i<patches.size(); ++i )
	{
		surfaceVariance[i] = patches[i].lambda0;
	}

// 	double variance = 0.0;
// 	for ( int i=0; i<patches.size(); ++i )
// 	{
// 		variance += ( patches[i].lambda0 - avg ) * ( patches[i].lambda0 - avg );
// 	}
// 	double sigma = sqrt( variance / patches.size() );

	std::vector<double> surfaceVarianceSort = surfaceVariance;
	double medianValue = meadian( surfaceVarianceSort );
	std::vector<double>().swap( surfaceVarianceSort );

	std::vector<double> absDiff( patches.size() );
	for( int i = 0; i < patches.size(); ++i )
	{
		absDiff[i] = fabs( surfaceVariance[i] - medianValue );
	}
	double MAD = a * meadian( absDiff );
	std::vector<double>().swap( absDiff );

 	double surfaceVarianceMin = medianValue;
 	double surfaceVarianceMax = medianValue + 2.5 * MAD ;

// 	double surfaceVarianceMin = 0.00040591;
// 	double surfaceVarianceMax = 0.00364508;
// 	//cout<<surfaceVarianceMin1<<"  "<<surfaceVarianceMax1<<endl;
 	cout<<surfaceVarianceMin<<"  "<<surfaceVarianceMax<<endl;

	double k = ( maxAngle - minAngle ) / ( surfaceVarianceMax - surfaceVarianceMin );
	std::vector<double> thAngle( patches.size(), 0.0 );
	for ( int i=0; i<patches.size(); ++i )
	{
		if ( patches[i].lambda0 < surfaceVarianceMin )
		{
			thAngle[i] = minAngle;
		}
		else if ( patches[i].lambda0 > surfaceVarianceMax )
		{
			thAngle[i] = maxAngle;
		}
		else
		{
			thAngle[i] = ( patches[i].lambda0 - surfaceVarianceMin ) * k + minAngle;
		}
	}

	// find the cluster for each data point
	std::vector<int> clusterIdx( this->pointNum, -1 );
	for ( int i=0; i<numPatch; ++i )
	{
		for ( int j=0; j<patches[i].idxIn.size(); ++j )
		{
			int idx = patches[i].idxIn[j];
			clusterIdx[idx] = i;
		}
	}

	// find the adjacent patches
	std::vector<std::vector<int> > patchAdjacent( numPatch );
#pragma omp parallel for
	for ( int i=0; i<numPatch; ++i )
	{
		std::vector<int> patchAdjacentTemp;
		std::vector<std::vector<int> > pointAdjacentTemp;
		for ( int j=0; j<patches[i].idxIn.size(); ++j )
		{
			int idx = patches[i].idxIn[j];
			for ( int m=0; m<pcaInfos[idx].idxIn.size(); ++m )
			{
				// in the idxIn of each other
				int idxPoint = pcaInfos[idx].idxIn[m];

				bool isIn = false;
				for ( int n=0; n<pcaInfos[idxPoint].idxIn.size(); ++n )
				{
					if ( pcaInfos[idxPoint].idxIn[n] == idx )
					{
						isIn = true;
					}
				}
				if ( ! isIn )
				{
					continue;
				}

				// accept the patch as a neighbor
				int idxPatch = clusterIdx[idxPoint];
				if ( idxPatch != i && idxPatch >=0 )
				{
					bool isIn = false;
					int n = 0;
					for ( n=0; n<patchAdjacentTemp.size(); ++n )
					{
						if ( patchAdjacentTemp[n] == idxPatch )
						{
							isIn = true;
							break;
						}
					}

					if ( isIn )
					{
						pointAdjacentTemp[n].push_back( idxPoint );
					}
					else
					{
						patchAdjacentTemp.push_back( idxPatch );

						std::vector<int> temp;
						temp.push_back( idxPoint );
						pointAdjacentTemp.push_back( temp );
					}
				}
			}
		}

		// repetition removal
		for ( int j=0; j<pointAdjacentTemp.size(); ++j )
		{
			std::vector<int> pointTemp;
			for ( int m=0; m<pointAdjacentTemp[j].size(); ++m )
			{
				bool isIn = false;
				for ( int n=0; n<pointTemp.size(); ++n )
				{
					if ( pointTemp[n] == pointAdjacentTemp[j][m] )
					{
						isIn = true;
						break;
					}
				}

				if ( !isIn )
				{
					pointTemp.push_back( pointAdjacentTemp[j][m] );
				}
			}

			if ( pointTemp.size() >= 3 )
			{
				patchAdjacent[i].push_back( patchAdjacentTemp[j] );
			}
		}
	}

	// plane merging
	std::vector<int> mergedIndex( numPatch, 0 );
	for ( int i=0; i<numPatch; ++i )
	{
		if ( !mergedIndex[i] )
		{
			int idxStarter = i;
			cv::Matx31d normalStarter = patches[idxStarter].normal;
			cv::Matx31d ptStarter = patches[idxStarter].planePt;
			double thAngleStarter = thAngle[idxStarter];

			std::vector<int> seedIdx;
			std::vector<int> patchIdx;
			seedIdx.push_back( idxStarter );
			patchIdx.push_back( idxStarter );

			int count = 0;
			while ( count < seedIdx.size() )
			{
				int idxSeed = seedIdx[count];
				cv::Matx31d normalSeed = patches[idxSeed].normal;
				cv::Matx31d ptSeed = patches[idxSeed].planePt;
				double thAngleSeed = thAngle[idxSeed];

				for ( int j=0; j<patchAdjacent[idxSeed].size(); ++j )
				{
					int idxCur = patchAdjacent[idxSeed][j];
					if ( mergedIndex[idxCur] )
					{
						continue;
					}

					cv::Matx31d normalCur = patches[idxCur].normal;
					cv::Matx31d ptCur = patches[idxCur].planePt;
					
					// plane angle deviation and distance
					double devAngle = 0.0;
					double devDis = 0.0;
					double thDev = 0.0;
					if ( this->planeMode == PLANE )
					{
						cv::Matx31d ptVector = ptCur - ptStarter;
						devAngle = acos( normalStarter.val[0] * normalCur.val[0] + normalStarter.val[1] * normalCur.val[1] + normalStarter.val[2] * normalCur.val[2] );
						devDis = abs( normalStarter.val[0] * ptVector.val[0] + normalStarter.val[1] * ptVector.val[1] + normalStarter.val[2] * ptVector.val[2] );
					}
					else
					{
						cv::Matx31d ptVector = ptCur - ptSeed;
						devAngle = acos( normalSeed.val[0] * normalCur.val[0] + normalSeed.val[1] * normalCur.val[1] + normalSeed.val[2] * normalCur.val[2] );
						devDis = abs( normalStarter.val[0] * ptVector.val[0] + normalStarter.val[1] * ptVector.val[1] + normalStarter.val[2] * ptVector.val[2] );
					}

					
					//if ( min( devAngle, fabs( CV_PI - devAngle ) ) < thAngle && devDis < thDev )
					//if ( min( devAngle, fabs( CV_PI - devAngle ) ) < thAngle && devDis < 1.0 )
					if ( min( devAngle, fabs( CV_PI - devAngle ) ) < min( thAngleSeed, thAngleStarter ) )
					{
						seedIdx.push_back( idxCur );
						patchIdx.push_back( idxCur );
						mergedIndex[idxCur] = 1;
					}
				}

				count ++;
			}

			// create a new cluster
			std::vector<int> patchNewCur;
			for ( int j=0; j<patchIdx.size(); ++j )
			{
				int idx = patchIdx[j];

				for ( int m=0; m<patches[idx].idxAll.size(); ++m )
				{
					patchNewCur.push_back( patches[idx].idxAll[m] );
				}
			}

			// 
			clusters.push_back( patchNewCur );
		}
	}

	cout<<"final plane's number: "<< clusters.size()<<endl;
}

double NLinkage::meadian( std::vector<double> &dataset )
{
	std::sort( dataset.begin(), dataset.end(), []( const double& lhs, const double& rhs ){ return lhs < rhs; } );

	return dataset[dataset.size()/2];
}