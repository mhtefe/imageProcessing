#ifndef _CONNECTED_COMPONENT_
#define _CONNECTED_COMPONENT_

#include "helpers.h"

// CONNECTED COMPONENT

static const int CONN_COMP_NBOR = 4;
static const int CONN_COMP_MORPH_MASK_SIZE = 3;
static const int CONN_COMP_MAX_OBJ_COUNT = 5;

static const int CONN_COMP_MIN_SIZE_1X_WIDTH = 6;
static const int CONN_COMP_MIN_SIZE_1X_HEIGHT = 6;
static const int CONN_COMP_MAX_SIZE_1X_WIDTH = 200;
static const int CONN_COMP_MAX_SIZE_1X_HEIGHT = 200;
static const int CONN_COMP_VELOCITY_X = 8;
static const int CONN_COMP_VELOCITY_Y = 4;
static const int CONN_COMP_ROI_WIDTH = 809;
static const int CONN_COMP_ROI_HEIGHT = 493;
static const int CONN_COMP_ROI_OFFSET_X = 0;
static const int CONN_COMP_ROI_OFFSET_Y = 0;

const int FILTER_PROCESS_BUFFER_COUNT = 5;

typedef struct M_Pair
{
	M_Pair() {}
	M_Pair(int _a, int _b) { a = _a; b = _b; }

	int a;
	int b;
} M_Pairs;


typedef struct M_Point
{
	M_Point() {}
	M_Point(int _x, int _y) { x = _x; y = _y; }

	int x;
	int y;
} M_Point;

class Component
{
private:
	bool mIgnored;
	M_Point mTopLeft;
	M_Point mBottomRight;

public:
	Component(M_Point p);

	void AddPoint(M_Point p);
	cv::Rect GetBoundingBox();
	M_Point GetCentroid();
	void ExpandWithBoundingBox(cv::Rect rect);
	void SetIgnored() { mIgnored = true; }
	bool IsIgnored() { return mIgnored; }
	bool IsIntersect(cv::Rect rect);
	int GetBoxArea() const;
};

typedef std::vector<Component> ComponentList;
typedef std::vector<M_Pair> PairList;

class ConnectedComponent 
{

public:

	ConnectedComponent();
	~ConnectedComponent();
	void process(Mat m);

	std::vector<cv::Rect> m_Objs[FILTER_PROCESS_BUFFER_COUNT];
private:

	int m_BufInd;

	cv::Mat m_ObjImages[FILTER_PROCESS_BUFFER_COUNT];

	int *labels;
	int *eqTable;
	int *cIndex;
	ComponentList componentList;
	ComponentList componentListSend[FILTER_PROCESS_BUFFER_COUNT];
	PairList pairList;
};

#endif