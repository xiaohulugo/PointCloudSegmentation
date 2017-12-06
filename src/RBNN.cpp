#include "RBNN.h"
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include "highgui.h"

using namespace std;

RBNN::RBNN( double voxelSize, double r, int nMin )
{
	this->voxelSize = voxelSize;
	this->r = r;
	this->nMin = nMin;
}

RBNN::~RBNN()
{
}

void RBNN::run( std::vector<std::vector<int> > &clusters )
{
	// create the voxels
	cout<<" creating the voxels ..."<<endl;
	double xMin = 100000000.0, xMax = -1000000000.0;
	double yMin = 100000000.0, yMax = -1000000000.0;
	double zMin = 100000000.0, zMax = -1000000000.0;
	for ( int i =0; i<this->pointNum; ++i )
	{
		if ( this->pointData[i][0] < xMin ) { xMin = this->pointData[i][0]; }
		if ( this->pointData[i][0] > xMax ) { xMax = this->pointData[i][0]; }

		if ( this->pointData[i][1] < yMin ) { yMin = this->pointData[i][1]; }
		if ( this->pointData[i][1] > yMax ) { yMax = this->pointData[i][1]; }

		if ( this->pointData[i][2] < zMin ) { zMin = this->pointData[i][2]; }
		if ( this->pointData[i][2] > zMax ) { zMax = this->pointData[i][2]; }
	}
	int numX = ( xMax - xMin ) / this->voxelSize + 1;
	int numY = ( yMax - yMin ) / this->voxelSize + 1;
	int numZ = ( zMax - zMin ) / this->voxelSize + 1;

	std::vector<std::vector<std::vector<std::vector<int> > > > voxels( numX, numY );
	for ( int i=0; i<numX; ++i )
	{
		for ( int j=0; j<numY; ++j )
		{
			voxels[i][j] = std::vector<std::vector<int> > ( numZ );
		}
	}

	std::vector<cv::Point3i> idxVoxel( this->pointNum );
	for ( int i=0; i<this->pointNum; ++i )
	{
		int vx = ( this->pointData[i][0] - xMin ) / this->voxelSize;
		int vy = ( this->pointData[i][1] - yMin ) / this->voxelSize;
		int vz = ( this->pointData[i][2] - zMin ) / this->voxelSize;

		voxels[vx][vy][vz].push_back( i );
		idxVoxel[i] = cv::Point3i( vx, vy, vz );
	}
	cout<<" the voxels done"<<endl;

	//
	std::vector<int> classIdx( this->pointNum, -1 );
	for ( int i=0; i<this->pointNum; ++i )
	{
		if ( i % 10000 == 0 )
		{
			cout<<i<<endl;
		}

		if ( classIdx[i] >= 0 )
		{
			continue;
		}
		int idx_i = i;

		std::vector<int> NN;
		findNeighborInRadius( voxels, idx_i, idxVoxel[idx_i], NN );

		for ( int j=0; j<NN.size(); ++j )
		{
			int idx_j = NN[j];

			if ( classIdx[idx_i] >=0 && classIdx[idx_j] >=0 )
			{
				if ( classIdx[idx_i] != classIdx[idx_j]  )
				{
					// merge clusters
					for ( int m =0; m<clusters[classIdx[idx_i]].size(); ++m )
					{
						int idx_ii = clusters[classIdx[idx_i]][m];
						classIdx[idx_ii] = classIdx[idx_j];

						clusters[classIdx[idx_j]].push_back( idx_ii );
					}
					clusters[classIdx[idx_i]].clear();
				}
			}
			else
			{
				if ( classIdx[idx_j] >=0 )
				{
					classIdx[idx_i] = classIdx[idx_j];
				}
				else
				{
					if ( classIdx[idx_i] >= 0 )
					{
						classIdx[idx_j] = classIdx[idx_i];
					}
				}
			}
		}

		if ( ! classIdx[idx_i] >= 0 )
		{
			std::vector<int> clusterNew;
			clusterNew.push_back( idx_i );
			classIdx[idx_i] = clusters.size();

			for ( int j=0; j<NN.size(); ++j )
			{
				int idx_j = NN[j];
				clusterNew.push_back( idx_j );
				classIdx[idx_j] = clusters.size();
			}

			clusters.push_back( clusterNew );
		}
	}

	//
	std::vector<std::vector<int> > clusterNew;
	for ( int i=0; i<clusters.size(); ++i )
	{
		if ( clusters[i].size() > this->nMin )
		{
			clusterNew.push_back( clusters[i] );
		}
	}
	clusters = clusterNew;

	cout<<" number of clusters : "<<clusters.size()<<endl;
}

void RBNN::setData( ANNpointArray pointData, int pointNum )
{
	this->pointData = pointData;
	this->pointNum = pointNum;
}


void RBNN::findNeighborInRadius( std::vector<std::vector<std::vector<std::vector<int> > > > &voxels, int idxPt, cv::Point3i &idxVoxel, std::vector<int> &idxNeib )
{
	int step = int ( this->r / this->voxelSize + 0.5 ); 
	int numX = voxels.size();
	int numY = voxels[0].size();
	int numZ = voxels[0][0].size();

	for ( int i=idxVoxel.x - step; i<= idxVoxel.x + step; ++i )
	{
		if ( i < 0 || i >= numX )
		{
			continue;
		}

		for ( int j=idxVoxel.y - step; j<= idxVoxel.y + step; ++j )
		{
			if ( j < 0 || j >= numY )
			{
				continue;
			}

			for ( int k=idxVoxel.z - step; k<= idxVoxel.z + step; ++k )
			{
				if ( k < 0 || k >= numZ )
				{
					continue;
				}

				//
				for ( int m=0; m<voxels[i][j][k].size(); ++m )
				{
					int idxCur = voxels[i][j][k][m];
					double dx = this->pointData[idxPt][0] - this->pointData[idxCur][0];
					double dy = this->pointData[idxPt][1] - this->pointData[idxCur][1];
					double dz = this->pointData[idxPt][2] - this->pointData[idxCur][2];

					double dis = sqrt( dx * dx + dy * dy + dz * dz );
					if ( dis < r )
					{
						idxNeib.push_back( idxCur );
					}
				}
			}
		}
	}
}
