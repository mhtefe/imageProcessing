#include "helpers.h"

int main()
{
	//these're some actual keypoints, collected using SIFT
	std::vector<cv::Point2d> im1Points;
	im1Points.push_back(Point2d(122, 126)); // x-y; col-row
	im1Points.push_back(Point2d(185, 49));
	im1Points.push_back(Point2d(327, 28));
	im1Points.push_back(Point2d(220, 311));

	std::vector<cv::Point2d> im2Points;
	im2Points.push_back(Point2d(95, 116));
	im2Points.push_back(Point2d(137, 55));
	im2Points.push_back(Point2d(228, 47));
	im2Points.push_back(Point2d(146, 285));

	Mat h = findHomography(im1Points, im2Points, RANSAC);
	Mat h2 = homography_dlt(im1Points, im2Points);
	int idx = 3;
	
	Point2f pt = mht_sourceToDestination(Point2f(im1Points[idx].x, im1Points[idx].y), h);
	Point2f pt2 = mht_sourceToDestination(Point2f(im1Points[idx].x, im1Points[idx].y), h2);

	// compare an old fashioned least squares solution with opencv function.
	// they gonna have same results
	Mat hamh = mht_affine_leastSquares(im1Points, im2Points);
	Mat hacv = estimateAffine2D(im1Points, im2Points);

	return 0;
}