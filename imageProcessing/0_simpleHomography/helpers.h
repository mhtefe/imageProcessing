#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>

// OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/video/tracking.hpp>

#ifdef HAVE_OPENCV_NONFREE
#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >=4
#include <opencv2/nonfree/gpu.hpp>
#include <opencv2/nonfree/features2d.hpp>
#endif
#endif
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#endif

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 50;
const float GOOD_MATCH_PERCENT = 0.15f;

// align images -> Satya Mallick, 
//https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

//these are good notes:
//Two images of a scene are related by a homography under two conditions.
//The two images are that of a plane(e.g.sheet of paper, credit card etc.).
//The two images were acquired by rotating the camera about its optical axis.We take such images while generating panoramas.
void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h)

{
	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	cvtColor(im1, im1Gray, CV_BGR2GRAY);
	cvtColor(im2, im2Gray, CV_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());


	// Draw top matches
	Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	imwrite("matches.jpg", imMatches);


	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	// Find homography
	h = findHomography(points1, points2, RANSAC);

	// Use homography to warp image
	warpPerspective(im1, im1Reg, h, im2.size());

}

Point2f mht_sourceToDestination(Point2f pt, Mat h, bool giveMessage = false)
{
	Mat po(3, 1, CV_64FC1);
	po.at<double>(0) = pt.x;
	po.at<double>(1) = pt.y;
	po.at<double>(2) = 1;

	Mat po1 = h * po;

	// homogenous to cartesian
	po1.at<double>(0) = po1.at<double>(0) / po1.at<double>(2);
	po1.at<double>(1) = po1.at<double>(1) / po1.at<double>(2);

	if(giveMessage)
		cout << "conf: " << 1 - abs(1 - po1.at<double>(2)) << endl;

	return Point2f(po1.at<double>(0), po1.at<double>(1));
}

Point2f mht_destinationToSource(Point2f pt, Mat h, bool giveMessage = false)
{
	return mht_sourceToDestination(pt, h.inv(), giveMessage);
}

template <class T>
T mht_euclidian_distance(T x0, T y0, T x1, T y1) 
{
	T result;
	result = pow( (x0-x1), 2) + pow((y0 - y1), 2);
	return ( sqrt(result));
}

#endif