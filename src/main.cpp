#include <stdio.h>
#include <vector>
#include "PointCloudSegmentation.h"
#include "cv.h"
#include "highgui.h"


using namespace cv;
using namespace std;

void main()
{
	std::string fileData = "J:\\data.txt";
	std::string fileResult = "C:\\clusterPoints.txt";
	ALGORITHM algorithm = PLINKAGE;  // PLINKAGE  REGIONGROW  SMOOTHCONSTRAINT  RBNNN

	PointCloudSegmentation segmenter;
	std::vector<std::vector<int> > clusters;
	segmenter.run(  fileData, algorithm, clusters );
	segmenter.writeOutClusters( fileResult, clusters );
}
