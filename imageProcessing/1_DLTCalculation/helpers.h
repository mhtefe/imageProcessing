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

//Eric.Marchand and these guys have a solution in here
// flow is simple; 
// build the linear solution matrix, 
// give them to SVD, 
// take the the smallest row of (vt)
// reshape it 3x3
// well the interesting thing is, they didn't applied any kind of normalization and hence their solution is pretty decent

// check this link:
// http://people.rennes.inria.fr/Eric.Marchand/pose-estimation/tutorial-pose-dlt-planar-opencv.html

cv::Mat homography_dlt(const std::vector< cv::Point2d > &x1, const std::vector< cv::Point2d > &x2)
{
	int npoints = (int)x1.size();
	cv::Mat A(2 * npoints, 9, CV_64F, cv::Scalar(0));
	// We need here to compute the SVD on a (n*2)*9 matrix (where n is
	// the number of points). if n == 4, the matrix has more columns
	// than rows. The solution is to add an extra line with zeros
	if (npoints == 4)
		A.resize(2 * npoints + 1, cv::Scalar(0));
	// Since the third line of matrix A is a linear combination of the first and second lines
	// (A is rank 2) we don't need to implement this third line
	for (int i = 0; i < npoints; i++) {					  // Update matrix A using eq. 33
		A.at<double>(2 * i, 3) = -x1[i].x;                // -xi_1
		A.at<double>(2 * i, 4) = -x1[i].y;                // -yi_1
		A.at<double>(2 * i, 5) = -1;                      // -1
		A.at<double>(2 * i, 6) = x2[i].y * x1[i].x;       //  yi_2 * xi_1
		A.at<double>(2 * i, 7) = x2[i].y * x1[i].y;       //  yi_2 * yi_1
		A.at<double>(2 * i, 8) = x2[i].y;                 //  yi_2
		A.at<double>(2 * i + 1, 0) = x1[i].x;             //  xi_1
		A.at<double>(2 * i + 1, 1) = x1[i].y;             //  yi_1
		A.at<double>(2 * i + 1, 2) = 1;                   //  1
		A.at<double>(2 * i + 1, 6) = -x2[i].x * x1[i].x;  // -xi_2 * xi_1
		A.at<double>(2 * i + 1, 7) = -x2[i].x * x1[i].y;  // -xi_2 * yi_1
		A.at<double>(2 * i + 1, 8) = -x2[i].x;            // -xi_2
	}
	// Add an extra line with zero.
	if (npoints == 4) {
		for (int i = 0; i < 9; i++) {
			A.at<double>(2 * npoints, i) = 0;
		}
	}
	cv::Mat w, u, vt;
	cv::SVD::compute(A, w, u, vt);
	double smallestSv = w.at<double>(0, 0);
	unsigned int indexSmallestSv = 0;
	for (int i = 1; i < w.rows; i++) {
		if ((w.at<double>(i, 0) < smallestSv)) {
			smallestSv = w.at<double>(i, 0);
			indexSmallestSv = i;
		}
	}
	cv::Mat h = vt.row(indexSmallestSv);
	if (h.at<double>(0, 8) < 0) // tz < 0
		h *= -1;
	cv::Mat _2H1(3, 3, CV_64F);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			_2H1.at<double>(i, j) = h.at<double>(0, 3 * i + j);
	return _2H1;
}


cv::Mat mht_affine_leastSquares(const std::vector< cv::Point2d > &x1, const std::vector< cv::Point2d > &x2)
{
	// let's change the linear system. Affine transformation matrix supposed to have shape of 2x3
	// and it's linear system consists of combinations of the given point pairs
	int npoints = (int)x1.size();

	cv::Mat A(2 * npoints, 6, CV_64F, cv::Scalar(0));
	cv::Mat C(2 * npoints, 1, CV_64F, cv::Scalar(0));
	// We need here to compute the SVD on a (n*2)*6 matrix (where n is
	// the number of points). if n == 3, the matrix has more columns
	// than rows. The solution is to add an extra line with zeros

	for (int i = 0; i < npoints; i++) 
	{	
		A.at<double>(2 * i, 0) = x1[i].x;   
		A.at<double>(2 * i, 1) = x1[i].y;   
		A.at<double>(2 * i, 2) = 1;         
		A.at<double>(2 * i, 3) = 0;       
		A.at<double>(2 * i, 4) = 0;       
		A.at<double>(2 * i, 5) = 0;         
		A.at<double>(2 * i + 1, 0) = 0;     
		A.at<double>(2 * i + 1, 1) = 0;     
		A.at<double>(2 * i + 1, 2) = 0;      
		A.at<double>(2 * i + 1, 3) = x1[i].x;
		A.at<double>(2 * i + 1, 4) = x1[i].y;
		A.at<double>(2 * i + 1, 5) = 1;     

		C.at<double>(2 * i, 0)     = x2[i].x;
		C.at<double>(2 * i + 1, 0) = x2[i].y;
	}

	Mat h = (A.t()*A).inv() * A.t() * C;
	h = h.reshape(1, 2);
	return h;
}

#endif