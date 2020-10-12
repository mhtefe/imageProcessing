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

#include "Eigen/Eigen"
#include <opencv2/core/eigen.hpp>

#include "cvplot/figure.h"

class AutoFocus
{
public:
	AutoFocus();
	~AutoFocus();
	float BayesMeasure(cv::Mat img);
	void InitializeDCT();

	int delta_w;
	int m_x;
	int m_y;
protected:
private:

	int m_lowF;
	int m_highF;
	int m_winRow;
	int m_winCol;

	
	Eigen::MatrixXf m_img;
	cv::Size m_roi;

	Eigen::MatrixXf block8x8;
	Eigen::MatrixXf block8x8T;
};

#endif