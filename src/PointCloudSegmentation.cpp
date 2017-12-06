#include "PointCloudSegmentation.h"
#include <fstream>
#include <omp.h>
#include "highgui.h"

#include "PCAFunctions.h"
#include "NLinkage.h"
#include "RBNN.h"
#include "RegionGrow.h"
#include "SmoothConstraint.h"

using namespace std;
using namespace cv;

PointCloudSegmentation::PointCloudSegmentation()
{
}

PointCloudSegmentation::~PointCloudSegmentation()
{
}

void PointCloudSegmentation::run( std::string filepath, ALGORITHM algorithm, std::vector<std::vector<int> > &clusters )
{
	if ( ! setDataFromFile( filepath ) )
	{
		cout<<"Data reading error! "<<endl;
	}

	if ( algorithm == PLINKAGE )             // Algorithm 1 : P-Linkage Segmentation 
	{
		int k = 100;

		bool outlierRemoval = true;
		std::vector<PCAInfo> pcaInfos;
		PCAFunctions pcaer;
		//pcaer.HalfK_PCA( this->pointData, this->pointNum, k, pcaInfos, outlierRemoval );
		pcaer.MCS_PCA( this->pointData, this->pointNum, k, pcaInfos, outlierRemoval );		
		
		double theta = 90.0 / 180.0 * CV_PI;
		PLANE_MODE planeMode = SURFACE;               // PLANE  SURFACE
		PCA_MODE pcaMode = ORIPCA;                    // RDPCA  ORIPCA
		NLinkage nlinkage( k, theta, planeMode, pcaMode );
		nlinkage.setData( this->pointData, this->pointNum, pcaInfos );
		nlinkage.run( clusters );
	}
	else if ( algorithm == REGIONGROW )    // Algorithm 2 : Region Growing 
	{
		int k = 100;

		bool outlierRemoval = true;
		std::vector<PCAInfo> pcaInfos;
		PCAFunctions pcaer;
		//pcaer.Ori_PCA( this->pointData, this->pointNum, k, pcaInfos, outlierRemoval );
		pcaer.MCS_PCA( this->pointData, this->pointNum, k, pcaInfos, outlierRemoval );		
		
		double theta = 5.0 / 180.0 * CV_PI;
		int RMin = 10;
		RegionGrow regiongrow( theta, RMin );
		regiongrow.setData( this->pointData, this->pointNum, pcaInfos );
		regiongrow.run( clusters );
	}
	else if ( algorithm == SMOOTHCONSTRAINT )
	{
		int k = 50;

		bool outlierRemoval = false;
		std::vector<PCAInfo> pcaInfos;
		PCAFunctions pcaer;
		pcaer.Ori_PCA( this->pointData, this->pointNum, k, pcaInfos, outlierRemoval );

		double theta = 5.0 / 180.0 * CV_PI;
		double percent = 0.75;
		SmoothConstraint smoothconstraint( theta, percent );
		smoothconstraint.setData( this->pointData, this->pointNum, pcaInfos );
		smoothconstraint.run( clusters );
	}
	else if ( algorithm == RBNNN )
	{
		double voxelSize = 1.0;
		double r = 2.5;
		int nMin = 100;
		RBNN rbnn( voxelSize, r, nMin );
		rbnn.setData( this->pointData, this->pointNum );
		rbnn.run( clusters );
	}
	else
	{
		cout<<" please choose an algorithm "<<endl;
	}
}

bool PointCloudSegmentation::setDataFromFile( std::string filepath )
{
	int i, j;
	int step = 1;

	// read in point data
	std::ifstream ptReader( filepath );
	std::vector<cv::Point3d> lidarPoints;
	double x = 0, y = 0, z = 0, color = 0;
	int a = 0, b = 0, c = 0; 
	int label = 0;
	int count = 0;
	if( ptReader.is_open() )
	{
		while ( !ptReader.eof() ) 
		{
			ptReader >> x >> y >> z >> a >> b >> c >> label;
			//ptReader >> x >> y >> z;
			//ptReader >> x >> y >> z >> color;
			if ( count == step )
			{
				lidarPoints.push_back( cv::Point3d( x, y, z ) );
				count = 0;
			}
			
			count ++;
		}
		ptReader.close();
	}
	else
	{
		return false;
	}

	// data processing
	this->pointNum = (int) lidarPoints.size();
	this->pointData = annAllocPts( this->pointNum, 3 );
	for ( int i = 0 ; i < this->pointNum ; i++ )
	{
		this->pointData[i][0] = (double)lidarPoints[i].x;
		this->pointData[i][1] = (double)lidarPoints[i].y;
		this->pointData[i][2] = (double)lidarPoints[i].z;
	}
	std::vector<cv::Point3d>().swap(lidarPoints);

	std::cout << "Total num of points: " << this->pointNum << "\n";

	return true;
}

void PointCloudSegmentation::setDataFromANN( ANNpointArray pointData, int pointNum )
{
	this->pointData = pointData;
	this->pointNum = pointNum;
}

void PointCloudSegmentation::writeOutClusters( string filePath, std::vector<std::vector<int> > &clusters )
{
	int i, j;

	std::vector<cv::Scalar> colors(30);
	colors[4] = cv::Scalar( 0, 0, 255 );
	colors[1] = cv::Scalar( 0, 255, 0 );
	colors[2] = cv::Scalar( 255, 0, 0 );
	colors[3] = cv::Scalar( 0, 0, 116 );
	colors[0] = cv::Scalar( 34,139,34 );
	colors[5] = cv::Scalar( 18,153,255 );
	colors[6] = cv::Scalar( 226, 43, 138 );

	for ( i=0; i<30; ++i )
	{
		int R = rand() % 255;
		int G = rand() % 255;
		int B = rand() % 255;

		colors[i] = cv::Scalar( B, G, R );
	}

	FILE *fp = fopen( filePath.c_str(), "w");
	for ( i=0; i<clusters.size(); ++i )
		//for ( i=0; i<1; ++i )
	{

		int R = rand() % 255;
		int G = rand() % 255;
		int B = rand() % 255;

		if ( ( R == 110 && G == 43 && B == 174 ) )
		{
			cout<<"123   "<<i<<endl;
		}

		for ( j=0; j<clusters[i].size(); ++j )
		{
			int idx = clusters[i][j];

			fprintf( fp, "%.3lf%15.3lf%15.3lf%10d%10d%10d\n", pointData[idx][0], pointData[idx][1], pointData[idx][2], R, G, B );
		}
	}

	fclose( fp );
}

void PointCloudSegmentation::writeOutClusters2( string filePath, std::vector<std::vector<int> > &clusters )
{
	int i, j;

	std::vector<int> indexCluster( pointNum, -1 );
	for ( i=0; i<clusters.size(); ++i )
	{
		for ( j=0; j<clusters[i].size(); ++j )
		{
			int idx = clusters[i][j];
			indexCluster[idx] = i;
		}
	}

	//
	std::vector<cv::Scalar> colors(clusters.size());
	for ( i=0; i<clusters.size(); ++i )
	{
		int R = rand() % 255;
		int G = rand() % 255;
		int B = rand() % 255;
		colors[i] = cv::Scalar( R, G, B );
	}

	// 
// 	int cols = 2549;
// 	int rows = 1083;

// 	int cols = 2530;
// 	int rows = 1083;

	int cols = 2500;
	int rows = 1076;

	cv::Mat img( rows, cols, CV_8UC3, cv::Scalar(0,0,0) );
	uchar* ptr = (uchar*) img.data;

	int count = 0;
	for ( i=0; i<rows; ++i )
	{
		for ( j=0; j<cols; ++j )
		{
			int idxCluster = indexCluster[count];
			if ( idxCluster >= 0 )
			{
				ptr[0] = colors[idxCluster].val[0];
				ptr[1] = colors[idxCluster].val[1];
				ptr[2] = colors[idxCluster].val[2];
			}

			count++;
			ptr += 3;
		}
	}
	imwrite( filePath, img );

}
