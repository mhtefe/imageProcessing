#include <boost/numeric/ublas/io.hpp>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <assert.h> 
#include <numeric>
#include "base/Tiler.hpp"
#include "TilerTestSuiteTestData.hpp"

// THIS HAS TO BE IFDEF CGI then use this, else just include cvTile.hpp
#ifdef HAVE_CGI
#include "../src/base/cvTileConversion.hpp"
#else
#include "base/cvTile.hpp"
#endif


using namespace std;
using namespace cvt;

cv::Size2i rasterSize(5, 5);
cv::Size2i tileSize(3, 3);
cv::Size2i testImageSize(768, 768);
int bandCount = 1;
int rowCount = 2;
int columnCount = 2;
int tileCount = 4;
int nodataValue = -1;
int bufferWidth = 1;

Tiler mt;
Tiler mtw;

string inputFilename1024("mosaic-1024-test.tif");
string inputFilename("mosaic-test-5x5.ntf");
string scratchFilename("tdata.mosaic-test-5x5.ntf");
string driverName("NITF");

#define TS_ASSERT assert
#define TS_ASSERT_EQUALS(a,b) assert(a == b)

/*
void testCreateZombie()
{
	cvTile<double> t;

	TS_ASSERT(t.getSize() == cv::Size2i(0, 0));
	TS_ASSERT(t.getBandCount() == 0);

	TS_ASSERT(t.getSize().height == t.getROI().height);
	TS_ASSERT(t.getSize().width == t.getROI().width);
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);
}

void testCreationBySizeAndBandCount()
{
	cv::Size2i s(1, 2);
	cvt::cvTile<double> t(s, 3);

	TS_ASSERT(t.getSize() == s);
	TS_ASSERT(t.getBandCount() == 3);

	TS_ASSERT(t.getSize() == t.getROI().size());
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);
}

void testCloneTile()
{
	cv::Size2i s(2, 1);
	cvTile<double> t(s, 3);

	TS_ASSERT(true == t.setMetadata("foo", "bar"));
	TS_ASSERT(true == t.setMetadata("zzz", "yyy"));
	TS_ASSERT("bar" == t.getMetadata("foo"));
	TS_ASSERT("" == t.getMetadata("brian"));
	TS_ASSERT(true == t.setROI(cv::Rect(cv::Point2i(1, 0), cv::Size2i(1, 1))));

	t.setNoDataValue(34.0);
	TS_ASSERT(34.0 == t.getNoDataValue());

	// CREATE THE CLONE
	cvTile<double> clone = t.cloneWithoutData(17);

	// TEST Band count of  Clone
	TS_ASSERT(17 == clone.getBandCount());


	// TEST Metadata Clone
	std::set<std::string> keys = clone.getMetadataKeys();
	TS_ASSERT(keys.size() == 2);
	TS_ASSERT(keys.find("foo") != keys.end());
	TS_ASSERT(keys.find("zzz") != keys.end());
	TS_ASSERT("bar" == clone.getMetadata("foo"));

	// TEST size count of  Clone
	TS_ASSERT(clone.getSize() == s);

	// ROI
	TS_ASSERT(cv::Size2i(1, 1) == clone.getROI().size());
	TS_ASSERT(1 == clone.getROI().x);
	TS_ASSERT(0 == clone.getROI().y);

	// No Data Value
	TS_ASSERT(34.0 == clone.getNoDataValue());
}

void testClonetestCreationByMatrixTileWithoutDataTo()
{
	std::cout << "Next test" << std::endl;
	cv::Size2i s(2, 1);
	cvTile<double> t(s, 3);

	TS_ASSERT(true == t.setMetadata("foo", "bar"));
	TS_ASSERT(true == t.setMetadata("zzz", "yyy"));
	TS_ASSERT("bar" == t.getMetadata("foo"));
	TS_ASSERT("" == t.getMetadata("brian"));
	TS_ASSERT(true == t.setROI(cv::Rect(cv::Point2i(1, 0), cv::Size2i(1, 1))));

	t.setNoDataValue(34.0);
	TS_ASSERT(34.0 == t.getNoDataValue());

	// CREATE THE CLONE
	cvTile<int> clone = t.cloneWithoutDataTo<int>(17);

	// TEST Band count of  Clone
	TS_ASSERT(17 == clone.getBandCount());

	// TEST Metadata Clone
	std::set<std::string> keys = clone.getMetadataKeys();
	TS_ASSERT(keys.size() == 2);
	TS_ASSERT(keys.find("foo") != keys.end());
	TS_ASSERT(keys.find("zzz") != keys.end());
	TS_ASSERT("bar" == clone.getMetadata("foo"));

	// TEST size count of  Clone
	TS_ASSERT(clone.getSize() == s);

	// ROI
	TS_ASSERT(cv::Size2i(1, 1) == clone.getROI().size());
	TS_ASSERT(1 == clone.getROI().x);
	TS_ASSERT(0 == clone.getROI().y);

	// No Data Value
	TS_ASSERT(34 == clone.getNoDataValue());
}

void testCloneSubsetOfTile()
{
	cv::Size2i s(2, 1);
	cvTile<double> t(s, 3);

	TS_ASSERT(true == t.setMetadata("foo", "bar"));
	TS_ASSERT(true == t.setMetadata("zzz", "yyy"));
	TS_ASSERT("bar" == t.getMetadata("foo"));
	TS_ASSERT("" == t.getMetadata("brian"));
	TS_ASSERT(true == t.setROI(cv::Rect(cv::Point2i(1, 0), cv::Size2i(1, 1))));

	t.setNoDataValue(34.0);
	TS_ASSERT(34.0 == t.getNoDataValue());

	TS_ASSERT(t.setBandName(0, std::string("band0")));
	TS_ASSERT(t.setBandName(1, std::string("band1")));
	TS_ASSERT(t.setBandName(2, std::string("band2")));

	// CREATE THE CLONE
	cvTile<double> clone = t.cloneSubset(0);

	// TEST Band count of  Clone
	TS_ASSERT(1 == clone.getBandCount());
	// TEST Band name of  Clone
	TS_ASSERT(clone.getBandName(0) == t.getBandName(0));
	TS_ASSERT(clone.getBandName(0) == std::string("band0"));

	// TEST Metadata Clone
	std::set<std::string> keys = clone.getMetadataKeys();
	TS_ASSERT(keys.size() == 2);
	TS_ASSERT(keys.find("foo") != keys.end());
	TS_ASSERT(keys.find("zzz") != keys.end());
	TS_ASSERT("bar" == clone.getMetadata("foo"));

	// TEST size count of  Clone
	TS_ASSERT(clone.getSize() == s);

	// ROI
	TS_ASSERT(cv::Size2i(1, 1) == clone.getROI().size());
	TS_ASSERT(1 == clone.getROI().x);
	TS_ASSERT(0 == clone.getROI().y);

	// No Data Value
	TS_ASSERT(34.0 == clone.getNoDataValue());
}

void testCreationBySizeAndBandCountWithInit()
{
	cv::Size2i s(1, 2);
	cvTile<double> t(s, 3, 77);

	TS_ASSERT(t.getSize() == s);
	TS_ASSERT(t.getBandCount() == 3);

	TS_ASSERT(t.getSize() == t.getROI().size());
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);

	for (int b = 0; b < t.getBandCount(); ++b)
	{
		for (int y = 0; y < t.getSize().height; ++y)
		{
			for (int x = 0; x < t.getSize().width; ++x)
			{
				//TS_ASSERT_DELTA(77, t[b].at<double>(y, x), 0.001);
			}
		}
	}
}

void testCreationByMatrix()
{
	cv::Mat m(5, 6, cv::DataType<double>::type);
	cvTile<double> t(m);

	TS_ASSERT(t.getSize() == cv::Size2i(6, 5));
	TS_ASSERT(t.getBandCount() == 1);

	TS_ASSERT(t.getSize() == t.getROI().size());
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);
}

void testCreationByVectorOfMatrices()
{
	vector< cv::Mat > v(3, cv::Mat(6, 7, cv::DataType<double>::type));
	cvTile<double> t(v);

	TS_ASSERT(t.getSize() == cv::Size2i(7, 6));
	TS_ASSERT(t.getBandCount() == 3);

	TS_ASSERT(t.getSize() == t.getROI().size());
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);
}

void testCreationByBufferSizeAndBandCount()
{
	vector<int> buffer1(9, 77);
	cvTile<int> t1(&buffer1[0], cv::Size2i(3, 3), 1);

	TS_ASSERT(t1.getSize() == cv::Size2i(3, 3));
	TS_ASSERT(t1.getBandCount() == 1);

	TS_ASSERT(t1.getSize() == t1.getROI().size());
	TS_ASSERT(0 == t1.getROI().x);
	TS_ASSERT(0 == t1.getROI().y);

	vector<int> buffer2(27, 77);
	cvTile<int> t2(&buffer2[0], cv::Size2i(3, 3), 3);

	TS_ASSERT(t2.getSize() == cv::Size2i(3, 3));
	TS_ASSERT(t2.getBandCount() == 3);

	TS_ASSERT(t2.getSize() == t2.getROI().size());
	TS_ASSERT(0 == t2.getROI().x);
	TS_ASSERT(0 == t2.getROI().y);
}

void testMetadata()
{
	cv::Size2i s(2, 1);
	cvTile<double> t(s, 3);

	TS_ASSERT(true == t.setMetadata("foo", "bar"));
	TS_ASSERT(true == t.setMetadata("zzz", "yyy"));
	TS_ASSERT("bar" == t.getMetadata("foo"));

	TS_ASSERT("" == t.getMetadata("brian"));

	std::set<std::string> keys = t.getMetadataKeys();
	TS_ASSERT(keys.size() == 2);
	TS_ASSERT(keys.find("foo") != keys.end());
	TS_ASSERT(keys.find("zzz") != keys.end());

	t.setMetadata("value", "33");
	TS_ASSERT_EQUALS(t.getMetadataAs<int>("value"), 33);
}

void testBandNames()
{
	cv::Size2i s(2, 1);
	cvTile<double> t(s, 3);

	TS_ASSERT(true == t.setBandName(0, "red"));
	TS_ASSERT(true == t.setBandName(1, "green"));
	TS_ASSERT(true == t.setBandName(2, "blue"));

	TS_ASSERT(0 == t.getBandIndex("red"));
	TS_ASSERT(1 == t.getBandIndex("green"));
	TS_ASSERT(2 == t.getBandIndex("blue"));

	TS_ASSERT("red" == t.getBandName(0));
	TS_ASSERT("green" == t.getBandName(1));
	TS_ASSERT("blue" == t.getBandName(2));

	TS_ASSERT("" == t.getBandName(3));
	TS_ASSERT("" == t.getBandName(-1));

	TS_ASSERT(0 > t.getBandIndex("ozy"));
}

void testROI()
{
	cv::Size2i s(10, 10);
	cvTile<double> t(s, 3);

	TS_ASSERT(t.getSize() == s);
	TS_ASSERT(t.getBandCount() == 3);

	TS_ASSERT(t.getSize() == t.getROI().size());
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);

	TS_ASSERT(true == t.setROI(cv::Rect(cv::Point2i(2, 2), cv::Size2i(6, 6))));

	TS_ASSERT(6 == t.getROI().size().height);
	TS_ASSERT(6 == t.getROI().size().width);

	TS_ASSERT(2 == t.getROI().x);
	TS_ASSERT(2 == t.getROI().y);
	cv::Rect old_roi = t.resetROI();

	TS_ASSERT_EQUALS(t.getROI(), cv::Rect(cv::Point2i(0, 0), s));

	TS_ASSERT_EQUALS(cv::Size2i(6, 6), old_roi.size());
	TS_ASSERT_EQUALS(2, old_roi.x);
	TS_ASSERT_EQUALS(2, old_roi.y);
}

void testExpandConstrictROI()
{
	cv::Size2i s(10, 10);
	cvTile<double> t(s, 3);

	TS_ASSERT(t.getSize() == s);
	TS_ASSERT(t.getBandCount() == 3);

	TS_ASSERT(t.getSize() == t.getROI().size());
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);

	TS_ASSERT(true == t.setROI(cv::Rect(cv::Point2i(2, 2), cv::Size2i(6, 6))));

	TS_ASSERT(6 == t.getROI().size().height);
	TS_ASSERT(6 == t.getROI().size().width);
	TS_ASSERT(2 == t.getROI().x);
	TS_ASSERT(2 == t.getROI().y);

	TS_ASSERT(t.expandROI(2));

	TS_ASSERT_EQUALS(t.getROI(), cv::Rect(s, cv::Point2i(0, 0)));

	TS_ASSERT_EQUALS(cv::Size2i(10, 10), t.getROI().size());
	TS_ASSERT_EQUALS(0, t.getROI().x);
	TS_ASSERT_EQUALS(0, t.getROI().y);

	// expand / constrict
	TS_ASSERT(t.constrictROI(2));
	TS_ASSERT(cv::Size2i(6, 6) == t.getROI().size());
	TS_ASSERT(2 == t.getROI().x);
	TS_ASSERT(2 == t.getROI().y);

	TS_ASSERT(!t.expandROI(3));
}

void testCropToROI()
{
	// create a 7x7 image with 3 bands
	cv::Size2i r(7, 7);
	cvTile<unsigned short> t(r, 3);

	// set the roi to be a 3x3 square in the middle of the image
	t.setROI(cv::Rect(cv::Point2i(2, 2), cv::Size2i(3, 3)));

	// initialize the image with some data
	for (int r = 0; r < 7; ++r)
		for (int c = 0; c < 7; ++c)
			t[0].at<unsigned short>(r, c) = r + c;

	// crop the tile to its ROI
	t.cropToROI();

	// validate the the size is correct
	TS_ASSERT(3 == t.getSize().height);
	TS_ASSERT(3 == t.getSize().width);

	// check some of the values to ensure that they are what we expect
	TS_ASSERT(4 == t[0].at<unsigned short>(0, 0));
	TS_ASSERT(5 == t[0].at<unsigned short>(1, 0));
	TS_ASSERT(8 == t[0].at<unsigned short>(2, 2));

	// ensure that the roi of the new image is correctly set to be the entire tile
	TS_ASSERT(cv::Size2i(3, 3) == t.getROI().size());
	TS_ASSERT(0 == t.getROI().x);
	TS_ASSERT(0 == t.getROI().y);
}

void testCopyCropToROI()
{
	// create a 7x7 image with 3 bands
	cv::Size2i r(7, 7);
	cvt::cvTile<unsigned short> t(r, 3);

	// set the roi to be a 3x3 square in the middle of the image
	t.setROI(cv::Rect(cv::Point2i(2, 2), cv::Size2i(3, 3)));

	// initialize the image with some data
	for (int r = 0; r < 7; ++r)
		for (int c = 0; c < 7; ++c)
			t[0].at<unsigned short>(r, c) = r + c;

	// crop the tile to its ROI
	cvt::cvTile<unsigned short> t2(t.copyCropToROI());

	// validate the the size is correct
	TS_ASSERT(3 == t2.getSize().height);
	TS_ASSERT(3 == t2.getSize().width);

	// check some of the values to ensure that they are what we expect
	TS_ASSERT(4 == t2[0].at<unsigned short>(0, 0));
	TS_ASSERT(5 == t2[0].at<unsigned short>(1, 0));
	TS_ASSERT(8 == t2[0].at<unsigned short>(2, 2));

	// ensure that the roi of the new image is correctly set to be the entire tile
	TS_ASSERT(cv::Size2i(3, 3) == t2.getROI().size());
	TS_ASSERT(0 == t2.getROI().x);
	TS_ASSERT(0 == t2.getROI().y);
}

void testConstMatrixByIndex()
{
	vector<int> buffer(12);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 1);
	const cvTile<int>& ct = t;
	const cv::Mat& m = ct[0];
	TS_ASSERT(3 == m.rows);
	TS_ASSERT(4 == m.cols);
	TS_ASSERT(0 == m.at<int>(0, 0));
	TS_ASSERT(4 == m.at<int>(1, 0));
	TS_ASSERT(11 == m.at<int>(2, 3));

	//TS_ASSERT_THROWS_ANYTHING(ct[1]);
}

void testMatrixByIndex()
{
	vector<int> buffer(12);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 1);
	cv::Mat& m = t[0];
	TS_ASSERT(3 == m.rows);
	TS_ASSERT(4 == m.cols);
	TS_ASSERT(0 == m.at<int>(0, 0));
	TS_ASSERT(4 == m.at<int>(1, 0));
	TS_ASSERT(11 == m.at<int>(2, 3));

	m.at<int>(0, 0) = 123;
	TS_ASSERT(123 == m.at<int>(0, 0));

	//TS_ASSERT_THROWS_ANYTHING(t[1]);
}

void testConstMatrixByName()
{
	vector<int> buffer(12);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 1);
	t.setBandName(0, "foo");
	const cvTile<int>& ct = t;
	const cv::Mat& m = ct["foo"];
	TS_ASSERT(3 == m.rows);
	TS_ASSERT(4 == m.cols);
	TS_ASSERT(0 == m.at<int>(0, 0));
	TS_ASSERT(4 == m.at<int>(1, 0));
	TS_ASSERT(11 == m.at<int>(2, 3));

	//TS_ASSERT_THROWS_ANYTHING(ct["bar"]);
}

void testMatrixByName()
{
	vector<int> buffer(12);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 1);
	t.setBandName(0, "foo");
	cv::Mat& m = t["foo"];
	TS_ASSERT(3 == m.rows);
	TS_ASSERT(4 == m.cols);
	TS_ASSERT(0 == m.at<int>(0, 0));
	TS_ASSERT(4 == m.at<int>(1, 0));
	TS_ASSERT(11 == m.at<int>(2, 3));

	m.at<int>(0, 0) = 123;
	TS_ASSERT(123 == m.at<int>(0, 0));

	//TS_ASSERT_THROWS_ANYTHING(t["bar"]);
}

void testSetMethod()
{
	cv::Size2i s(1, 2);
	cvTile<double> t(s, 3, 77);

	TS_ASSERT(t.getSize() == s);
	TS_ASSERT(t.getBandCount() == 3);

	t.set(66);

	for (int b = 0; b < t.getBandCount(); ++b)
	{
		for (int y = 0; y < t.getSize().height; ++y)
		{
			for (int x = 0; x < t.getSize().width; ++x)
			{
				//TS_ASSERT_DELTA(66, t[b].at<double>(y, x), 0.001);
			}
		}
	}
}

void testSetMethodWithMask()
{
	cv::Size2i s(3, 2);
	cvTile<double> t(s, 3, 77);

	TS_ASSERT(t.getSize() == s);
	TS_ASSERT(t.getBandCount() == 3);

	// recall that ublas matrices are (row, column) not (x, y)
	cv::Mat mask(2, 3, cv::DataType<bool>::type);
	mask.at<bool>(0, 0) = true;
	mask.at<bool>(0, 1) = true;
	mask.at<bool>(1, 2) = true;
	mask.at<bool>(0, 2) = false;
	mask.at<bool>(1, 0) = false;
	mask.at<bool>(1, 1) = false;

	t.set(66, mask);

	for (int b = 0; b < t.getBandCount(); ++b)
	{
		//TS_ASSERT_DELTA(t[b].at<double>(0, 0), 66, 0.001);
		//TS_ASSERT_DELTA(t[b].at<double>(0, 1), 66, 0.001);
		//TS_ASSERT_DELTA(t[b].at<double>(1, 2), 66, 0.001);

		//TS_ASSERT_DELTA(t[b].at<double>(0, 2), 77, 0.001);
		//TS_ASSERT_DELTA(t[b].at<double>(1, 0), 77, 0.001);
		//TS_ASSERT_DELTA(t[b].at<double>(1, 1), 77, 0.001);
	}
}

void testOperatorParenAccess()
{
	vector<int> buffer(27);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(3, 3), 3);

	TS_ASSERT(0 == t(0, 0)[0]);
	TS_ASSERT(9 == t(0, 0)[1]);
	TS_ASSERT(18 == t(0, 0)[2]);

	TS_ASSERT(1 == t(0, 1)[0]);
	TS_ASSERT(10 == t(0, 1)[1]);
	TS_ASSERT(19 == t(0, 1)[2]);
}

void testOperatorParenCast()
{
	vector<int> buffer(27);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(3, 3), 3);

	// ensure that the TileVectorProxy object returned by operator()
	// can be correctly implicitly cast to a vector
	vector<int> v = t(0, 1);
	TS_ASSERT(3 == v.size());
	TS_ASSERT(1 == v[0]);
	TS_ASSERT(10 == v[1]);
	TS_ASSERT(19 == v[2]);
}

void testOperatorParenUpdate()
{
	vector<int> buffer(27);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(3, 3), 3);

	// ensure that operator() can update the data correctly
	// and verify using both element access methods
	t(0, 1)[2] = 3;
	TS_ASSERT(3 == t(0, 1)[2]);
	TS_ASSERT(3 == t[2].at<int>(0, 1));

	// ensure that it doesn't matter which element access
	// method is used to update the data
	t[2].at<int>(0, 1) = -1;
	TS_ASSERT(-1 == t(0, 1)[2]);
	TS_ASSERT(-1 == t[2].at<int>(0, 1));
}

void testOperatorParenBounds()
{
	vector<int> buffer(27);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(3, 3), 3);

	// if BOOST_UBLAS_NDEBUG is defined, uBLAS is compiled in release mode and does
	// not do bounds checking
#ifndef BOOST_UBLAS_NDEBUG
			// warning: boost emits error messages in the following two cases
	//TS_ASSERT_THROWS_ANYTHING(t(9, 9));
	//TS_ASSERT_THROWS_ANYTHING(t(-1, -1));
#endif
}

void testConstOperatorParenAccess()
{
	vector<int> buffer(27);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(3, 3), 3);

	const cvTile<int>& ct = t;

	TS_ASSERT(0 == ct(0, 0)[0]);
	TS_ASSERT(9 == ct(0, 0)[1]);
	TS_ASSERT(18 == ct(0, 0)[2]);

	TS_ASSERT(1 == ct(0, 1)[0]);
	TS_ASSERT(10 == ct(0, 1)[1]);
	TS_ASSERT(19 == ct(0, 1)[2]);

#ifdef COMPILE_TIME_TEST
	// the following should fail to compile
	ct(0, 0)[0] = 3;
#endif
}

void testConstOperatorParenCast()
{
	vector<int> buffer(27);
	iota(buffer.begin(), buffer.end(), 0);
	const cvTile<int> ct(&buffer[0], cv::Size2i(3, 3), 3);

	// ensure that the TileVectorProxy object returned by operator()
	// can be correctly and implicitly cast to a vector
	vector<int> v = ct(0, 1);
	TS_ASSERT(3 == v.size());
	TS_ASSERT(1 == v[0]);
	TS_ASSERT(10 == v[1]);
	TS_ASSERT(19 == v[2]);
}

void testOperatorParenSelfAssignment()
{
	vector<int> buffer(27);

	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(3, 3), 3);
	const cvTile<int>& ct = t;

	// precondition
	TS_ASSERT(0 == t(0, 0)[0]);
	TS_ASSERT(9 == t(0, 0)[1]);
	TS_ASSERT(18 == t(0, 0)[2]);

	TS_ASSERT(1 == t(0, 1)[0]);
	TS_ASSERT(10 == t(0, 1)[1]);
	TS_ASSERT(19 == t(0, 1)[2]);

	TS_ASSERT(2 == t(0, 2)[0]);
	TS_ASSERT(11 == t(0, 2)[1]);
	TS_ASSERT(20 == t(0, 2)[2]);

	//
	// special case: assignment to same location
	//

	// assignment
	t(0, 0) = t(0, 0);

	// postcondition
	TS_ASSERT(0 == t(0, 0)[0]);
	TS_ASSERT(9 == t(0, 0)[1]);
	TS_ASSERT(18 == t(0, 0)[2]);

	//
	// test non-const -> non-const
	//

	// assignment
	t(0, 0) = t(0, 1);

	// postcondition
	TS_ASSERT(1 == t(0, 0)[0]);
	TS_ASSERT(10 == t(0, 0)[1]);
	TS_ASSERT(19 == t(0, 0)[2]);

	//
	// test const -> non-const
	//

	// assignment
	t(0, 0) = ct(0, 2);

	// postcondition
	TS_ASSERT(2 == t(0, 0)[0]);
	TS_ASSERT(11 == t(0, 0)[1]);
	TS_ASSERT(20 == t(0, 0)[2]);

#ifdef COMPILE_TIME_TEST
	// the following should fail to compile

	//
	// test non-const -> const
	//

	ct(0, 0) = t(0, 0);

	//
	// test const -> const
	//

	ct(0, 0) = ct(0, 0);
#endif
}

void testOperatorParenInteroperability()
{
	vector<int> buffer(27);

	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t1(&buffer[0], cv::Size2i(3, 3), 3);
	const cvTile<int>& ct1 = t1;

	iota(buffer.begin(), buffer.end(), 1);
	cvTile<int> t2(&buffer[0], cv::Size2i(3, 3), 3);
#ifdef COMPILE_TIME_TEST
	const cvTile<int>& ct2 = t2;
#endif

	// precondition
	TS_ASSERT(1 == t2(0, 0)[0]);
	TS_ASSERT(10 == t2(0, 0)[1]);
	TS_ASSERT(19 == t2(0, 0)[2]);

	//
	// test non-const -> non-const
	//

	// assignment
	t2(0, 0) = t1(0, 0);

	// postcondition
	TS_ASSERT(0 == t2(0, 0)[0]);
	TS_ASSERT(9 == t2(0, 0)[1]);
	TS_ASSERT(18 == t2(0, 0)[2]);

	//
	// test const -> non-const
	//

	// assignment
	t2(0, 0) = ct1(0, 2);

	// postcondition
	TS_ASSERT(2 == t2(0, 0)[0]);
	TS_ASSERT(11 == t2(0, 0)[1]);
	TS_ASSERT(20 == t2(0, 0)[2]);

#ifdef COMPILE_TIME_TEST
	// the following should fail to compile

	//
	// test non-const -> const
	//

	ct1(0, 0) = t2(0, 0);

	//
	// test const -> const
	//

	ct1(0, 0) = ct2(0, 0);
#endif

}

void testNoDataValue()
{
	vector<int> buffer(27);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(3, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(7));
	TS_ASSERT(7 == t.getNoDataValue());
}

void testROIValidMaskAll()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMask(cvt::valid_mask::ALL);
	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	// check that the pixel that we set to have the nodata value in any band
	TS_ASSERT(false == validMask.at<bool>(0, 1));

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 0) || (j != 1))
				TS_ASSERT(true == validMask.at<bool>(i, j));
}

void testROIValidMaskAny()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// set this pixel to have the nodata value in all bands
	t[0].at<int>(1, 2) = 1;
	t[1].at<int>(1, 2) = 1;
	t[2].at<int>(1, 2) = 1;

	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMask(cvt::valid_mask::ANY);
	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	// check that the pixel that we set to have the nodata value in all bands
	TS_ASSERT(false == validMask.at<bool>(1, 2));

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 1) || (j != 2))
				TS_ASSERT(true == validMask.at<bool>(i, j));
}

void testROIValidMaskMajority()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// set this pixel to have the nodata value in all bands
	t[0].at<int>(1, 2) = 1;
	t[1].at<int>(1, 2) = 1;
	t[2].at<int>(1, 2) = 1;

	// set this pixel to have the nodata value in 2 of the 3 bands
	t[0].at<int>(2, 0) = 1;
	t[1].at<int>(2, 0) = 1;


	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMask(cvt::valid_mask::MAJORITY);
	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	// check that the pixel that we set to have the nodata value in a majority of the bands
	TS_ASSERT(false == validMask.at<bool>(1, 2));
	TS_ASSERT(false == validMask.at<bool>(2, 0));

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if (((i != 1) || (j != 2)) && ((i != 2) || (j != 0)))
				TS_ASSERT(true == validMask.at<bool>(i, j));
}

void testROIValidMaskByIndex()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	cv::Mat validMask = t.getValidMask(0);
	TS_ASSERT(false == validMask.at<bool>(0, 1));

	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 0) || (j != 1))
				TS_ASSERT(true == validMask.at<bool>(i, j));

	//TS_ASSERT_THROWS_ANYTHING(t.getValidMask(9));
	//TS_ASSERT_THROWS_ANYTHING(t.getValidMask(-1));
}

void testROIValidMaskByName()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);
	t.setBandName(0, "foo");

	TS_ASSERT(true == t.setNoDataValue(1));

	cv::Mat validMask = t.getValidMask("foo");
	TS_ASSERT(false == validMask.at<bool>(0, 1));

	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 0) || (j != 1))
				TS_ASSERT(true == validMask.at<bool>(i, j));

	//TS_ASSERT_THROWS_ANYTHING(t.getValidMask("bar"));
}


void testValidMaskAll()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMask(cvt::valid_mask::ALL);
	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	// check that the pixel that we set to have the nodata value in any band
	TS_ASSERT(false == validMask.at<bool>(0, 1));

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 0) || (j != 1))
				TS_ASSERT(true == validMask.at<bool>(i, j));
}

void testGetMethodOnMatrix()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	for (int i = 0; i < 3; i++)
	{
		for (int ii = 0; ii < t[i].rows; ++ii)
		{
			//I need to put some data into my vector...
			for (int jj = 0; jj < t[i].cols; ++jj)
			{
				//std::cout << t.get(t[i],ii,jj) << std::endl;
			}
		}
	}
}

void testValidMaskAny()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// set this pixel to have the nodata value in all bands
	t[0].at<int>(1, 2) = 1;
	t[1].at<int>(1, 2) = 1;
	t[2].at<int>(1, 2) = 1;

	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMask(cvt::valid_mask::ANY);
	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	// check that the pixel that we set to have the nodata value in all bands
	TS_ASSERT(false == validMask.at<bool>(1, 2));

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 1) || (j != 2))
				TS_ASSERT(true == validMask.at<bool>(i, j));

}

void testValidMaskMajority()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// set this pixel to have the nodata value in all bands
	t[0].at<int>(1, 2) = 1;
	t[1].at<int>(1, 2) = 1;
	t[2].at<int>(1, 2) = 1;

	// set this pixel to have the nodata value in 2 of the 3 bands
	t[0].at<int>(2, 0) = 1;
	t[1].at<int>(2, 0) = 1;


	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMask(cvt::valid_mask::MAJORITY);
	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	// check that the pixel that we set to have the nodata value in a majority of the bands
	TS_ASSERT(false == validMask.at<bool>(1, 2));
	TS_ASSERT(false == validMask.at<bool>(2, 0));

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if (((i != 1) || (j != 2)) && ((i != 2) || (j != 0)))
				TS_ASSERT(true == validMask.at<bool>(i, j));
}

void testValidMaskByIndex()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	cv::Mat validMask = t.getValidMask(0);
	TS_ASSERT(false == validMask.at<bool>(0, 1));

	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 0) || (j != 1))
				TS_ASSERT(true == validMask.at<bool>(i, j));

	//TS_ASSERT_THROWS_ANYTHING(t.getValidMask(9));
	//TS_ASSERT_THROWS_ANYTHING(t.getValidMask(-1));
}

void testValidMaskByName()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);
	t.setBandName(0, "foo");

	TS_ASSERT(true == t.setNoDataValue(1));

	cv::Mat validMask = t.getValidMask("foo");
	TS_ASSERT(false == validMask.at<bool>(0, 1));

	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 0) || (j != 1))
				TS_ASSERT(true == validMask.at<bool>(i, j));

	//TS_ASSERT_THROWS_ANYTHING(t.getValidMask("bar"));
}

void testMaskWithoutNodataValue()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	cv::Mat mask(3, 4, cv::DataType<unsigned char>::type, cv::Scalar(255));
	mask.at<unsigned char>(0, 1) = 0;

	TS_ASSERT_EQUALS(t.setMask(mask), true);

	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMask(cvt::valid_mask::ALL);

	TS_ASSERT_EQUALS(validMask.rows, 3);
	TS_ASSERT_EQUALS(validMask.cols, 4);

	// check that the pixel that we set to have the nodata value in any band
	TS_ASSERT_EQUALS(validMask.at<bool>(0, 1), false);

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
			if ((i != 0) || (j != 1))
				TS_ASSERT_EQUALS(validMask.at<bool>(i, j), true);
}

void testValueValidMaskAll()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMaskByValue<short>(cvt::valid_mask::ALL, 30);
	TS_ASSERT(3 == validMask.rows);
	TS_ASSERT(4 == validMask.cols);

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
		{
			if ((i != 0) || (j != 1))
			{
				TS_ASSERT(30 == validMask.at<short>(i, j));
			}
			else
			{
				TS_ASSERT(0 == validMask.at<short>(i, j));
			}
		}
}

void testValueValidMaskAny()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT_EQUALS(true, t.setNoDataValue(1));

	// set this pixel to have the nodata value in all bands
	t[0].at<int>(1, 2) = 1;
	t[1].at<int>(1, 2) = 1;
	t[2].at<int>(1, 2) = 1;

	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMaskByValue<unsigned char>(cvt::valid_mask::ANY, 255);
	TS_ASSERT_EQUALS(3, validMask.rows);
	TS_ASSERT_EQUALS(4, validMask.cols);

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
		{
			if ((i != 1) || (j != 2))
			{
				TS_ASSERT(255 == validMask.at<unsigned char>(i, j));
			}
			else
			{
				TS_ASSERT(0 == validMask.at<unsigned char>(i, j));
			}
		}
}

void testValueValidMaskMajority()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	TS_ASSERT(true == t.setNoDataValue(1));

	// set this pixel to have the nodata value in all bands
	t[0].at<int>(1, 2) = 1;
	t[1].at<int>(1, 2) = 1;
	t[2].at<int>(1, 2) = 1;

	// set this pixel to have the nodata value in 2 of the 3 bands
	t[0].at<int>(2, 0) = 1;
	t[1].at<int>(2, 0) = 1;


	// get the valid mask and validate its size
	cv::Mat validMask = t.getValidMaskByValue(cvt::valid_mask::MAJORITY, true);
	TS_ASSERT_EQUALS(3, validMask.rows);
	TS_ASSERT_EQUALS(4, validMask.cols);

	// check that the pixel that we set to have the nodata value in a majority of the bands
	TS_ASSERT_EQUALS(false, validMask.at<bool>(1, 2));
	TS_ASSERT_EQUALS(false, validMask.at<bool>(2, 0));

	// now check all the other pixels
	for (int i = 0; i < validMask.rows; ++i)
		for (int j = 0; j < validMask.cols; ++j)
		{
			if (((i != 1) || (j != 2)) && ((i != 2) || (j != 0)))
			{
				TS_ASSERT_EQUALS(true, validMask.at<bool>(i, j));
			}
			else
			{
				TS_ASSERT_EQUALS(false, validMask.at<bool>(i, j));
			}
		}
}

void testValueMaskWithoutNodataValue()
{
	vector<int> buffer(36);
	iota(buffer.begin(), buffer.end(), 0);
	cvTile<int> t(&buffer[0], cv::Size2i(4, 3), 3);

	cv::Mat mask = cv::Mat::zeros(3, 4, cv::DataType<unsigned char>::type);
	mask.at<unsigned char>(0, 1) = 255; // except for this little guy

	TS_ASSERT_EQUALS(mask.at<unsigned char>(0, 0), 0);
	TS_ASSERT_EQUALS(mask.at<unsigned char>(0, 1), 255);

	TS_ASSERT_EQUALS(t.setMask(mask), true);

	cv::Mat validMask = t.getValidMaskByValue<unsigned char>(cvt::valid_mask::ALL, 42);
	TS_ASSERT_EQUALS(validMask.rows, 3);
	TS_ASSERT_EQUALS(validMask.cols, 4);

	TS_ASSERT_EQUALS((std::count(validMask.begin<unsigned char>(), validMask.end<unsigned char>(), 42)), validMask.rows * validMask.cols);
}

void xtestMatrixLogicalNot()
{
	//This test uses matrix_utility.hpp
	//which only deals with boost.
	//not sure if this is needed
}

void xtestOverwriteMetadata()
{
	//REMOVED THIS TEST BECAUSE IT WAS USING tile_utility.hpp from Tile
	//and it was testing that functionality of that function.
}

void xtestNondestructiveCopyMetadata()
{
	//REMOVED THIS TEST BECAUSE IT WAS USING tile_utility.hpp from Tile
	//and it was testing that functionality of that function.
}

void testCorrectTypeEvaluation()
{
	{
		cvt::cvTile<unsigned int> unsigned_int;
		TS_ASSERT(unsigned_int.getType() == 7);

		cvt::cvTile<bool> boolean_val;
		TS_ASSERT(boolean_val.getType() == 0);

		cvt::cvTile<unsigned short> unsigned_short;
		TS_ASSERT(unsigned_short.getType() == 2);

		cvt::cvTile<short> signed_short;
		TS_ASSERT(signed_short.getType() == 3);

		cvt::cvTile<int> signed_int;
		TS_ASSERT(signed_int.getType() == 4);

		cvt::cvTile<char> normal_char;
		TS_ASSERT(normal_char.getType() == 1);

		cvt::cvTile<unsigned char> unsigned_char;
		TS_ASSERT(unsigned_char.getType() == 0);

		cvt::cvTile<long> normal_long;
		TS_ASSERT(normal_long.getType() == 7);

		cvt::cvTile<float> normal_float;
		TS_ASSERT(normal_float.getType() == 5);

		cvt::cvTile<double> normal_double;
		TS_ASSERT(normal_double.getType() == 6);
	}
}
*/
void testCreateFromOriginalFileUsing2DindexAnd256x256Tile()
{
	Tiler read_tiler;
	std::string sourceFile("F:/Demo/Istanbul_Test16_1/RGB/1/image_std3.tif");
	std::string outFile("F:/Demo/Istanbul_Test16_1/RGB/1/test2-3.tif");

	TS_ASSERT(NoError == read_tiler.open(sourceFile));

	//tile Size
	const cv::Size2i sz(256, 256);

	//set the cvTileSize so the image will be partitioned based on this
	read_tiler.setCvTileSize(sz);

	Tiler write_tiler;
	//set the cvTileSize for the writer image so it knows how the tiles are to be put back
	write_tiler.setCvTileSize(read_tiler.getCvTileSize());

	if (boost::filesystem::exists(outFile)) {
		boost::filesystem::remove(outFile);
	}

	TS_ASSERT(NoError == write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), read_tiler.getBandCount(), Depth8U));

	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(0, 0), 0));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(0, 1), 1));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(0, 2), 2));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(1, 0), 3));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(1, 1), 4));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(1, 2), 5));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(2, 0), 6));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(2, 1), 7));
	TS_ASSERT(NoError == write_tiler.putCvTile(read_tiler.getCvTile2D<unsigned char>(2, 2), 8));

	write_tiler.close();
	read_tiler.close();
}


int main(int argc, char** argv)
{
	testCreateFromOriginalFileUsing2DindexAnd256x256Tile();
	//testSetMethod();
	//testBandNames();
	//Tiler read_tiler;0
	//std::string sourceFile("test2.tif");
	//std::string outFile("test2-4.tif");

	//read_tiler.open(sourceFile);

	////tile Size
	//const cv::Size2i sz(16, 16);

	////set the cvTileSize so the image will be partitioned based on this
	//read_tiler.setCvTileSize(sz);

	//Tiler write_tiler;
	////set the cvTileSize for the writer image so it knows how the tiles are to be put back
	//write_tiler.setCvTileSize(read_tiler.getCvTileSize());

	//if (boost::filesystem::exists(outFile)) {
	//	boost::filesystem::remove(outFile);
	//}

	// write_tiler.create(outFile, "GTiff", read_tiler.getRasterSize(), read_tiler.getBandCount(), Depth8U);

	////loop through all tiles and put them into the writer Tiler
	//for (int i = 0; i < read_tiler.getCvTileCount(); ++i)
	//{
	//	 write_tiler.putCvTile(read_tiler.getCvTile<unsigned char>(i), i);
	//}

	//write_tiler.close();
	//read_tiler.close();
	////////////////////////////////////////////////////////////////////////// //////////////////////////////////////////////////////////////////////////


	return 0;
}