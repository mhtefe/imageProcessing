#include "helpers.h"

int main(int argc, char** argv)
{
	cv::Mat im1 = cv::imread("rickRights.png");
	cv::Mat im2 = cv::imread("rickRightsWarped.png");

	// test upper function first
	cv::Mat h;
	cv::Mat imReg;
	alignImages(im2, im1, imReg, h);

	//these're some actual keypoints, collected using SIFT
	std::vector<cv::Point2f> im1Points;
	im1Points.push_back(Point2f(122, 126)); // x-y; col-row
	im1Points.push_back(Point2f(185, 49));
	im1Points.push_back(Point2f(327, 28));
	im1Points.push_back(Point2f(220, 311));

	std::vector<cv::Point2f> im2Points;
	im2Points.push_back(Point2f(95, 116));
	im2Points.push_back(Point2f(137, 55));
	im2Points.push_back(Point2f(228, 47));
	im2Points.push_back(Point2f(146, 285));

	/**
	 * test 1: im2 point to im1 point
	 */
	 // cartesian to homogenous
	int idx = 2;
	Point2f pt = sourceToDestination(im2Points[idx], h);

	cout <<idx << ": "<< pt.x << " - " << pt.y <<  endl;


	/**
	 * tes2: im1 point to im2 point
	 */
	idx = 0;
	pt = destinationToSource(im1Points[idx], h);
	
	cout << idx << ": " << pt.x << " - " << pt.y << endl;

	// for debug given points
	//for (int i = 0; i < im1Points.size(); i++)
	//{
	//	cv::circle(im1, im1Points[i], 8.0, cv::Scalar(0, 0, 255), 1, cv::FILLED);
	//	cv::circle(im2, im2Points[i], 8.0, cv::Scalar(255, 255, 0), 1, cv::FILLED);
	//}
	//cv::imshow("im1", im1);
	//cv::imshow("im2", im2);

	cv::waitKey(0);

	std::cout << "Hello World!\n";
}
