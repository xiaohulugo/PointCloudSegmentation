#include "RegionGrow.h"
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include "highgui.h"

using namespace std;

RegionGrow::RegionGrow( double theta, int Rmin )
{
	this->theta = theta;
	this->Rmin = Rmin;
}

RegionGrow::~RegionGrow()
{
}

void RegionGrow::run( std::vector<std::vector<int> > &clusters )
{
	double b = 1.4826;

	// sort the data points according to their curvature
	std::vector<std::pair<int,double> > idxSorted( this->pointNum );
	for ( int i=0; i<this->pointNum; ++i )
	{
		idxSorted[i].first = i;
		idxSorted[i].second = pcaInfos[i].lambda0;
	}
	std::sort( idxSorted.begin(), idxSorted.end(), [](const std::pair<int,double>& lhs, const std::pair<int,double>& rhs) { return lhs.second < rhs.second; } );

	// begin region growing
	std::vector<int> used( this->pointNum, 0 );
	for ( int i=0; i<this->pointNum; ++i )
	{
		if ( used[i] )
		{
			continue;
		}

		if ( i % 10000 == 0 )
		{
			cout<<i<<endl;
		}

		//
		std::vector<int> clusterNew;
		clusterNew.push_back( idxSorted[i].first );

		int count = 0;
		while( count < clusterNew.size() )
		{
			int idxSeed = clusterNew[count];
			int num = pcaInfos[idxSeed].idxIn.size();
			cv::Matx31d normalSeed = pcaInfos[idxSeed].normal;

			// EDth
			std::vector<double> EDs( num );
			for ( int j=0; j<num; ++j )
			{
				int idx = pcaInfos[idxSeed].idxIn[j];
				double dx = this->pointData[idxSeed][0] - this->pointData[idx][0];
				double dy = this->pointData[idxSeed][1] - this->pointData[idx][1];
				double dz = this->pointData[idxSeed][2] - this->pointData[idx][2];

				EDs[j] = sqrt( dx * dx + dy * dy + dz * dz );
			}
			std::sort( EDs.begin(), EDs.end(), [](const double& lhs, const double& rhs) { return lhs < rhs; } );
			double EDth = EDs[ EDs.size() / 2 ];

			// ODth
			cv::Matx31d h_mean( 0, 0, 0 );
			for( int j = 0; j < num; ++j )
			{
				int idx = pcaInfos[idxSeed].idxIn[j];
				h_mean += cv::Matx31d( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
			}
			h_mean *= ( 1.0 / num );

			std::vector<double> ODs( num );
			for( int j = 0; j < num; ++j )
			{
				int idx = pcaInfos[idxSeed].idxIn[j];
				cv::Matx31d pt( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
				cv::Matx<double, 1, 1> OD_mat = ( pt - h_mean ).t() * pcaInfos[idxSeed].normal;
				double OD = fabs( OD_mat.val[0] );
				ODs[j] = OD;
			}

			// calculate the Rz-score for all points using ODs
			std::vector<double> sorted_ODs( ODs.begin(), ODs.end() );
			double median_OD =  meadian( sorted_ODs );
			std::vector<double>().swap( sorted_ODs );

			std::vector<double> abs_diff_ODs( num );
			for( int j = 0; j < num; ++j )
			{
				abs_diff_ODs[j] = fabs( ODs[j] - median_OD );
			}
			double MAD_OD = b * meadian( abs_diff_ODs );
			double ODth = median_OD + 2.0 * MAD_OD;

			// point cloud collection
			for( int j = 0; j < num; ++j )
			{
				int idx = pcaInfos[idxSeed].idxIn[j];
				if ( used[idx] )
				{
					continue;
				}

				if ( ODs[j] < ODth && EDs[j] < EDth )
				{
					cv::Matx31d normalCur = pcaInfos[idx].normal;
					double angle = acos( normalCur.val[0] * normalSeed.val[0] + normalCur.val[1] * normalSeed.val[1] + normalCur.val[2] * normalSeed.val[2] );
					if ( min( angle, CV_PI -angle ) < this->theta )
					{
						clusterNew.push_back( idx );
						used[idx] = 1;
					}
				}
			}

			count ++;
		}

		if ( clusterNew.size() > this->Rmin )
		{
			clusters.push_back( clusterNew );
		}
	}

	cout<<" number of clusters : "<<clusters.size()<<endl;
}

void RegionGrow::setData( ANNpointArray pointData, int pointNum, std::vector<PCAInfo> &pcaInfos )
{
	this->pointData = pointData;
	this->pointNum = pointNum;
	this->pcaInfos = pcaInfos;
}

double RegionGrow::meadian( std::vector<double> &dataset )
{
	std::sort( dataset.begin(), dataset.end(), []( const double& lhs, const double& rhs ){ return lhs < rhs; } );

	return dataset[dataset.size()/2];
}