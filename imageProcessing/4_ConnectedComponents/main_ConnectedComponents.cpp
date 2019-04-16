#include "ConnectedComponents.h"

#include <ctype.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
	string img_file = "components.jpg";

	Mat input_image = imread(img_file);
	Mat image_to_proc;
	cv::cvtColor(input_image, image_to_proc, CV_BGR2GRAY);
	cv::threshold(image_to_proc, image_to_proc, 128, 255, CV_THRESH_BINARY_INV);

	ConnectedComponent c;
	c.process(image_to_proc);

	waitKey(0);
	return 0;
}