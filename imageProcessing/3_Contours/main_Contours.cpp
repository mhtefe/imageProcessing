#include "helpers.h"

#include <ctype.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
	string img_file = "primitive.jpg";

	Mat input_image = imread(img_file);

	Mat image_to_proc = input_image;
	//manually threshold image
	cv::cvtColor(image_to_proc, image_to_proc, CV_BGR2GRAY);
	cv::threshold(image_to_proc, image_to_proc, 128, 255, CV_THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	findContours(image_to_proc, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Draw contours
	Mat drawing = Mat::zeros(image_to_proc.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	Rect bounding_rect;
	Mat image_to_proc2;
	input_image.copyTo(image_to_proc2);
	for (int i = 0; i < contours.size(); i++)
	{
		bounding_rect = boundingRect(contours[i]);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(image_to_proc2, bounding_rect, color, 2, 8, 0);
	}
	 
	vector< vector<Point> > hull(contours.size());
	for (int i = 0; i < contours.size(); i++)
		convexHull(Mat(contours[i]), hull[i], false);

	Mat drawing2 = Mat::zeros(image_to_proc.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color_contours = Scalar(0, 255, 0);  
		Scalar color = Scalar(255, 255, 0);  
		// draw ith convex hull
		drawContours(drawing2, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}

	waitKey(0);

	return 0;
}