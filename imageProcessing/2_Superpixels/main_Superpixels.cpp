#include "helpers.h"

#include <ctype.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
	string img_file = "thai.jpg";

	// algorithm parameters
	int algorithm = SLICO;
	int region_size = 10;  
	int ruler = 100;  
	int min_element_size = 10; 
	int num_iterations = 10;  

	// read the input image
	Mat input_image;
	input_image = imread(img_file);
	if (input_image.empty())
	{
		cout << "Could not open image..." << img_file << "\n";
		return -1;
	}

	// segmentation mask and result map
	Mat result, mask;

	// define a frame (Mat) for pre processing 
	Mat frame;
	input_image.copyTo(frame);
	input_image.copyTo(result);
	
	// blur the image to remove unnecessary details, and change color space (it's suggested convert to CieLAB, but lab or HSV will work fine )
	blur(frame, frame, Size(3, 3));
	cvtColor(frame, frame, COLOR_BGR2Lab);

	// run the algorithm
	Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(frame, SLICO, region_size, float(ruler));
	slic->iterate(num_iterations);
	if (min_element_size > 0)
		slic->enforceLabelConnectivity(min_element_size);

	/**
	 * Now I want to take a closer look for some individual segments and get their bounding boxes
	 * beware; id's of the segments must be covered in the vector 'someSegments' if any other image is being used for segmentation
	 */
	Mat labels;
	slic->getLabels(labels);
	Mat seg_part;
	vector<int> someSegments{ 10, 100, 500, 1400,1401,1402,1403, 2000, 3000, 4000 };
	for (int i = 0; i < someSegments.size(); ++i)
	{
		Mat1b mask_for_seg = (labels == someSegments[i]);
		Rect bbox = boundingRect(mask_for_seg);
		
		input_image.copyTo(seg_part, mask_for_seg);
		rectangle(seg_part, bbox, Scalar(255, 0, 0));
	}

	// get the contours for displaying
	slic->getLabelContourMask(mask, true);
	result.setTo(Scalar(0, 0, 255), mask);

	imshow("All Segments", result);
	imshow("Some Segments", seg_part);

	int c = waitKey(0);

	return 0;
}