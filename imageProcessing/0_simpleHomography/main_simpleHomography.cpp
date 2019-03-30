#include "helpers.h"

int main(int argc, char** argv)
{
	cv::Mat im1 = cv::imread("rickRights.png");
	cv::Mat im2 = cv::imread("rickRightsWarped.png");

	// test upper function first
	cv::Mat h;
	cv::Mat imReg;
	alignImages(im1, im2, imReg, h);

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
	 * test 1:  
	 */
	 // cartesian to homogenous
	int idx = 2;
	Point2f pt = mht_sourceToDestination(im1Points[idx], h, true);
	cout << idx << ": " << im2Points[idx] << endl;
	cout <<idx << ": "<< pt<<  endl << endl;


	/**
	 * tes2: 
	 */
	idx = 0;
	pt = mht_destinationToSource(im2Points[idx], h, true);
	
	cout << idx << ": " << im1Points[idx] << endl;
	cout << idx << ": " << pt << endl << endl;

	/**
	 * test3:
	 */
	Mat im1Reg2;
	h = findHomography(im1Points, im2Points, RANSAC);
	
	// note that, 4 coplanar point pairs on a image plane will gives us perspective transformation matrix, which in this case is equal to homography
	// we can check that
	//Mat hp = getPerspectiveTransform(im1Points, im2Points);
	
	warpPerspective(im1, im1Reg2, h, im2.size());

	// check a boundry if it retruns a fine result
	pt = mht_sourceToDestination(Point2f(0, 0), h, true);
	cout << idx << ": " << pt << endl << endl;

	cv::waitKey(0);

	std::cout << "Hello World!\n";
}
