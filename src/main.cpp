#include <stdio.h>
#include <vector>
#include <fstream>

#include "nanoflann.hpp"
#include "utils.h"
#include "cv.h"

#include "PointGrowAngleOnly.h"
#include "PointGrowAngleDis.h"
#include "ClusterGrowPLinkage.h"

using namespace cv;
using namespace std;

void readDataFromFile(std::string filepath, PointCloud<double> &cloud)
{
	cloud.pts.reserve(10000000);
	cout << "Reading data ..." << endl;

	// 1. read in point data
	std::ifstream ptReader(filepath);
	std::vector<cv::Point3d> lidarPoints;
	double x = 0, y = 0, z = 0, color = 0;
	double nx, ny, nz;
	int a = 0, b = 0, c = 0;
	int labelIdx = 0;
	int count = 0;
	int countTotal = 0;
	if (ptReader.is_open())
	{
		while (!ptReader.eof())
		{
			//ptReader >> x >> y >> z >> a >> b >> c >> labelIdx;
			//ptReader >> x >> y >> z >> a >> b >> c >> color;
			//ptReader >> x >> y >> z >> color >> a >> b >> c;
			ptReader >> x >> y >> z >> a >> b >> c ;
			//ptReader >> x >> y >> z;
			//ptReader >> x >> y >> z >> color;
			//ptReader >> x >> y >> z >> nx >> ny >> nz;

			cloud.pts.push_back(PointCloud<double>::PtData(x, y, z));

		}
		ptReader.close();
	}

	std::cout << "Total num of points: " << cloud.pts.size() << "\n";
}


void writeOutClusters(string filePath, PointCloud<double> &pointData, std::vector<std::vector<int> > &clusters)
{
	std::vector<cv::Scalar> colors(30);
	colors[4] = cv::Scalar(0, 0, 255);
	colors[1] = cv::Scalar(0, 255, 0);
	colors[2] = cv::Scalar(255, 0, 0);
	colors[3] = cv::Scalar(0, 0, 116);
	colors[0] = cv::Scalar(34, 139, 34);
	colors[5] = cv::Scalar(18, 153, 255);
	colors[6] = cv::Scalar(226, 43, 138);
	for (int i = 0; i<30; ++i)
	{
		int R = rand() % 255;
		int G = rand() % 255;
		int B = rand() % 255;
		colors[i] = cv::Scalar(B, G, R);
	}

	FILE *fp = fopen(filePath.c_str(), "w");
	for (int i = 0; i<clusters.size(); ++i)
	{
		int R = rand() % 255;
		int G = rand() % 255;
		int B = rand() % 255;
		for (int j = 0; j<clusters[i].size(); ++j)
		{
			int idx = clusters[i][j];
			fprintf(fp, "%.3lf%15.3lf%15.3lf%10d%10d%10d\n", 
				pointData.pts[idx].x, pointData.pts[idx].y, pointData.pts[idx].z, R, G, B);
		}
	}

	fclose(fp);
}

void main()
{
	std::string fileData = "C:\\Users\\LXH\\Desktop\\HU005.txt";
	std::string fileResult = "C:\\Users\\LXH\\Desktop\\result.txt";

	// step1: read in data
	PointCloud<double> pointData;
	readDataFromFile(fileData, pointData);

	// step2: build kd-tree
	int k = 100;
	std::vector<PCAInfo> pcaInfos;
	PCAFunctions pcaer;
	pcaer.PCA(pointData, 100, pcaInfos);

	// step3: run point segmentation algorithm
	int algorithm = 0;
	std::vector<std::vector<int>> clusters;

	// Algorithm1: segmentation via PLinkage based clustering
	if (algorithm == 0) 
	{
		double theta = 90.0 / 180.0 * CV_PI;
		PLANE_MODE planeMode = SURFACE;               // PLANE  SURFACE
		ClusterGrowPLinkage segmenter(k, theta, planeMode);
		segmenter.setData(pointData, pcaInfos);
		segmenter.run(clusters);
	}
	// Algorithm2: segmentation via normal angle similarity
	else if (algorithm == 1)
	{
		double theta = 5.0 / 180.0 * CV_PI;
		double percent = 0.75;
		PointGrowAngleOnly segmenter(theta, percent);
		segmenter.setData(pointData, pcaInfos);
		segmenter.run(clusters);
	}
	// Algorithm3: segmentation via normal angle similarity and point-plane distance
	else
	{
		double theta = 10.0 / 180.0 * CV_PI;
		int RMin = 10;  // minimal number of points per cluster
		PointGrowAngleDis segmenter(theta, RMin);
		segmenter.setData(pointData, pcaInfos);
		segmenter.run(clusters);
	}
	
	// step4: write out result
	writeOutClusters(fileResult, pointData, clusters);
}
