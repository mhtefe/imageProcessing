#include "ConnectedComponents.h"
#include "DisjointSets.h"

Component::Component(M_Point p) : mIgnored(false)
{
	mTopLeft.x = mBottomRight.x = p.x;
	mTopLeft.y = mBottomRight.y = p.y;
}

void Component::AddPoint(M_Point p)
{
	if (p.x < mTopLeft.x)
	{
		mTopLeft.x = p.x;
	}
	else if (p.x > mBottomRight.x)
	{
		mBottomRight.x = p.x;
	}

	if (p.y < mTopLeft.y)
	{
		mTopLeft.y = p.y;
	}
	else if (p.y > mBottomRight.y)
	{
		mBottomRight.y = p.y;
	}
}

cv::Rect Component::GetBoundingBox()
{
	cv::Rect rect;

	rect.x = mTopLeft.x;
	rect.y = mTopLeft.y;
	rect.width = mBottomRight.x - mTopLeft.x + 1;
	rect.height = mBottomRight.y - mTopLeft.y + 1;

	return rect;
}

int Component::GetBoxArea() const
{
	return (mTopLeft.x - mBottomRight.x)*(mTopLeft.y - mBottomRight.y);
}

bool Component::IsIntersect(cv::Rect rect)
{
	bool result = false;

	if (mTopLeft.x >= rect.x && mTopLeft.y >= rect.y)
	{
		if (rect.x + rect.width >= mTopLeft.x && rect.y + rect.height >= mTopLeft.y)
		{
			result = true;
			return result;
		}
	}
	if (mTopLeft.x < rect.x && mTopLeft.y >= rect.y)
	{
		if (rect.y + rect.height >= mTopLeft.y && rect.x <= mBottomRight.x)
		{
			result = true;
			return result;
		}
	}
	if (mTopLeft.x >= rect.x && mTopLeft.y < rect.y)
	{
		if (rect.x + rect.width >= mTopLeft.x && rect.y <= mBottomRight.y)
		{
			result = true;
			return result;
		}
	}
	if (mTopLeft.x < rect.x && mTopLeft.y < rect.y)
	{
		if (rect.x <= mBottomRight.x && rect.y <= mBottomRight.y)
		{
			result = true;
			return result;
		}
	}

	return result;
}

M_Point Component::GetCentroid()
{
	M_Point p;

	p.x = (mTopLeft.x + mBottomRight.x) / 2;
	p.y = (mTopLeft.y + mBottomRight.y) / 2;

	return p;
}

void Component::ExpandWithBoundingBox(cv::Rect rect)
{
	if (rect.x < mTopLeft.x)
	{
		mTopLeft.x = rect.x;
	}

	if (rect.y < mTopLeft.y)
	{
		mTopLeft.y = rect.y;
	}

	if (rect.x + rect.width > mBottomRight.x)
	{
		mBottomRight.x = rect.x + rect.width;
	}

	if (rect.y + rect.height > mBottomRight.y)
	{
		mBottomRight.y = rect.y + rect.height;
	}
}


ConnectedComponent::ConnectedComponent()
{
	//	pTemp = NULL;
	labels = NULL;
	m_BufInd = 0;

	for (int i = 0; i < FILTER_PROCESS_BUFFER_COUNT; i++)
	{
		m_ObjImages[i]  = cv::Mat();
	}
}


ConnectedComponent::~ConnectedComponent()
{
	delete labels;
}




#define isComponent(val) (val==255)
#define isLabeled(val) (val!=0)
#define index(x,y,c) (y*c + x)

void ConnectedComponent::process(Mat pImage)
{
	int HEIGHT = pImage.rows;
	int WIDTH = pImage.cols;
	int channels = pImage.channels();
	int step = pImage.step;

	unsigned char *img = (unsigned char *)pImage.ptr();

	int *tmp;

	if (labels == NULL)
	{
		labels = new int[WIDTH*HEIGHT];
	}

	memset(labels, 0, WIDTH * HEIGHT * sizeof(int));

	int labelIndex = 0;

	ComponentList componentList;
	PairList pairList;

	int nbor = CONN_COMP_NBOR;
	int LabelU;
	int LabelL;

	tmp = labels;


	for (int j = 0; (j < HEIGHT); j++)
	{
		for (int i = 0; (i < WIDTH) ; i++, img++, labels++)
		{
			if (isComponent(*(img)))
			{
				LabelU = 0;
				LabelL = 0;

				for (int k = 1; k <= nbor; k++)
				{
					if (i - k >= 0)
					{
						if (isLabeled(*(labels - k)))
						{
							LabelL = *(labels - k);
							break;
						}
					}
				}

				for (int k = 1; k <= nbor; k++)
				{
					if (j - k >= 0)
					{
						if (isLabeled(*(labels - (k*WIDTH))))
						{
							LabelU = *(labels - (k*WIDTH));
							break;
						}
					}
				}

				if (isLabeled(LabelU))
				{
					*labels = LabelU;

					if (isLabeled(LabelL) && (LabelU != LabelL))
					{
						pairList.push_back(M_Pair(LabelL, LabelU));
					}
				}
				else if (isLabeled(LabelL))
				{
					*labels = LabelL;
				}
				else
				{
					labelIndex++;
					*labels = labelIndex;
				}
			}
		}

		img += (step - WIDTH);
	}

	labels = tmp;

	int rows = labelIndex + 1;
	int cols = labelIndex + 1;
	eqTable = new int[rows*cols];
	memset(eqTable, 0, rows * cols * sizeof(int));

	for (int i = 0; i < pairList.size(); i++)
	{

		eqTable[index(pairList[i].a, pairList[i].b, cols)] = 1;
		eqTable[index(pairList[i].b, pairList[i].a, cols)] = 1;
	}

	DisjointSets s(labelIndex + 1);

	int jj, ii;
	int m, n;

	for (ii = 2; ii < rows; ii++)
	{
		for (jj = 1; jj < ii; jj++)
		{
			if (eqTable[index(ii, jj, cols)] == 1)
			{
				m = s.FindSet(ii);
				n = s.FindSet(jj);
				if (m != n)
				{
					s.Union(m, n);
				}
			}
		}
	}

	cIndex = new int[labelIndex + 1];
	memset(cIndex, 0, (labelIndex + 1) * sizeof(int));
	int leybl;
	for (int j = 0; j < HEIGHT; j++)
	{
		for (int i = 0; i < WIDTH; i++, labels++)
		{
			if (*labels != 0)
			{
				leybl = s.FindSet(*labels);
				if (*labels != leybl)
				{
					*labels = leybl;
				}

				if (cIndex[*labels] == 0)
				{
					Component comp(M_Point(i, j));

					componentList.push_back(comp);

					cIndex[*labels] = componentList.size();
				}
				else
				{
					componentList[cIndex[*labels] - 1].AddPoint(M_Point(i, j));
				}
			}
		}
	}


	m_Objs[m_BufInd].clear();
	componentListSend[m_BufInd].clear();

	for (int i = 0; i < componentList.size(); i++)
	{
		if (!componentList[i].IsIgnored())
		{
			for (int j = 0; j < componentList.size(); j++)
			{
				if (j == i)
				{
					continue;
				}

				if (!componentList[j].IsIgnored())
				{
					if (componentList[j].IsIntersect(componentList[i].GetBoundingBox()))
					{
						componentList[j].SetIgnored();
						componentList[i].ExpandWithBoundingBox(componentList[j].GetBoundingBox());
					}
				}
			}
			if (componentList[i].GetBoundingBox().height >= CONN_COMP_MIN_SIZE_1X_HEIGHT || componentList[i].GetBoundingBox().width >= CONN_COMP_MIN_SIZE_1X_WIDTH)
			{
				m_Objs[m_BufInd].push_back(componentList[i].GetBoundingBox());
				componentListSend[m_BufInd].push_back(componentList[i]);
			}
		}
	}

	if (componentListSend[m_BufInd].size() > /*20*/CONN_COMP_MAX_OBJ_COUNT)
	{
		componentListSend[m_BufInd].clear();
	}

	if (m_Objs[m_BufInd].size() > CONN_COMP_MAX_OBJ_COUNT)
	{
		m_Objs[m_BufInd].clear();
	}

	for (int k = 0; k< FILTER_PROCESS_BUFFER_COUNT; ++k)
	{
		if (m_Objs[m_BufInd].size() != 0)
		{
			std::vector<cv::Rect> r = m_Objs[m_BufInd];
			if (r.size() > 0)
			{
				for (int i = 0; i < r.size(); ++i)
					cv::rectangle(pImage, r[i], cv::Scalar(255, 0, 0));
			}
		}
	}

	labels = tmp;

	delete eqTable;
	delete cIndex;

	return;
}

