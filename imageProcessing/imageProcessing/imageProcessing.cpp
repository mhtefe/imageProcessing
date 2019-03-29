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
	Mat po2(3, 1, CV_64FC1);
	po2.at<double>(0) = im2Points[idx].x;
	po2.at<double>(1) = im2Points[idx].y;
	po2.at<double>(2) = 1;

	Mat po1 = h * po2;

	// homogenous to cartesian
	po1.at<double>(0) = po1.at<double>(0) / po1.at<double>(2);
	po1.at<double>(1) = po1.at<double>(1) / po1.at<double>(2);

	cout << "original im1[" << idx << "]  x-y: " << im1Points[idx].x  << " - " << im1Points[idx].y << endl;
	cout << "calculat im1[" << idx << "]  x-y: " << po1.at<double>(0) << " - " << po1.at<double>(1) << endl;
	cout << "dist: " << mht_euclidian_distance<double>(im1Points[idx].x, im1Points[idx].y,
													   po1.at<double>(0), po1.at<double>(1)) << endl;
	cout << "conf: " << 1 - abs(1 - po1.at<double>(2)) << endl;

	cout << endl;


	/**
	 * tes2: im1 point to im2 point
	 */
	int idx2 = 1;
	Mat ppo1(3, 1, CV_64FC1);
	ppo1.at<double>(0) = im1Points[idx2].x;
	ppo1.at<double>(1) = im1Points[idx2].y;
	ppo1.at<double>(2) = 1;

	Mat ppo2 = h.inv() * ppo1;
	ppo2.at<double>(0) = ppo2.at<double>(0) / ppo2.at<double>(2);
	ppo2.at<double>(1) = ppo2.at<double>(1) / ppo2.at<double>(2);

	cout << "original im2[" << idx << "]  x-y: " << im2Points[idx2].x << " - " << im2Points[idx2].y << endl;
	cout << "calculat im2[" << idx << "]  x-y: " << ppo2.at<double>(0) << " - " << ppo2.at<double>(1) << endl;
	cout << "dist: " << mht_euclidian_distance<double>(im2Points[idx2].x, im2Points[idx2].y,
											            ppo2.at<double>(0), ppo2.at<double>(1)) << endl;
	cout << "conf: " <<  1 - abs( 1 -  ppo2.at<double>(2) ) <<endl ;

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
