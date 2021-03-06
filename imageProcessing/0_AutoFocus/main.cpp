#include "helpers.h"

using namespace std;

AutoFocus::AutoFocus()
{
	m_winRow = 8;
	m_winCol = 8;
	m_lowF = 0;
	m_highF = 6;
	m_x = 768 / 2 + 40; 
	m_y = 576 / 2 - 40; 
	delta_w = 82;

	m_img = Eigen::MatrixXf::Zero(delta_w, delta_w);
	m_roi.width = delta_w;
	m_roi.height = delta_w;

	InitializeDCT();
}

AutoFocus::~AutoFocus()
{

}

void AutoFocus::InitializeDCT()
{
	block8x8 = Eigen::MatrixXf::Zero(m_winRow, m_winCol);
	block8x8 << 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
		0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904,
		0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619,
		0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157,
		0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536,
		0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
		0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
		0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975;

	block8x8T = Eigen::MatrixXf::Zero(m_winRow, m_winCol);
	block8x8T = block8x8.transpose();

	uint8_t z;
}

float AutoFocus::BayesMeasure(cv::Mat img)
{
	if (img.channels() != 1)
	{
		return 0;
	}

	cv::Rect roi = cv::Rect(m_x, m_y, delta_w, delta_w);
	cv::cv2eigen(img(roi) , m_img);

	int measure = 0;
	int pp = 0;
	int pT = 3;
	float overlap = 1;
	Eigen::MatrixXf I = Eigen::MatrixXf::Zero(m_winRow, m_winCol);

	// simulasyonla ilgili sinir degerlerine dikkat et
	int x0, x1;
	for (int i = 0; i < m_winRow; ++i)
	{
		x0 = m_lowF - i + 1;
		if (x0 < 1)
			x0 = 1;

		x1 = m_highF - i + 1;
		if (x1 > m_winCol)
			x1 = m_winCol;

		for (int j = x0 - 1; j < x1; ++j)
		{
			I(i, j) = 1;
		}
	}

	float val = floor((float(m_roi.height) / float(m_winRow)) / overlap);
	int valCount = int(val);

	val = floor((float(m_roi.width) / float(m_winCol)) / overlap);
	int valCount2 = int(val);

	Eigen::VectorXf vecMean = Eigen::VectorXf::Constant(valCount*valCount2, -1);
	int counter = 0;

	int e, f;
	for (int i = 0; i < valCount; ++i)
	{
		float E = 0;
		e = i * m_winRow*overlap;
		if (e + m_winRow > m_roi.height)
			break;

		for (int j = 0; j < valCount2; ++j)
		{
			f = j * m_winCol*overlap;
			if (f + m_winCol > m_roi.width)
				break;

			Eigen::MatrixXf Z = m_img.block(e, f, m_winRow, m_winCol);
			Eigen::MatrixXf G = block8x8 * Z*block8x8T;

			// I daki valid degeleri al ve topla
			float sum = 0.0;
			for (int x = 0; x < m_winCol; ++x)
			{
				for (int y = 0; y < m_winRow; ++y)
				{
					if (I(y, x) == 1)
					{
						sum += abs(G(y, x));
					}
				}
			}

			sum = pow(sum, 2);

			if (sum == 0)
			{
				E = -1;
			}
			else
			{
				for (int x = 0; x < m_winCol; ++x)
				{
					for (int y = 0; y < m_winRow; ++y)
					{
						if (I(y, x) == 1)
						{
							E += pow(float(G(y, x)), 2);
						}
					}
				}

				E = E / sum;
			}

			vecMean(counter++) = E;
		}
	}

	float means = vecMean.mean();
	return (1 - means);
}

int main(int argc, char** argv)
{
	for (int i = 1; i < 17; ++i)
	{
		std::string filename = to_string(i) + ".png";

		cv::Mat sample = cv::imread(filename);
		cv::Mat sample_gray;
		cvtColor(sample, sample_gray, cv::COLOR_BGR2GRAY);

		AutoFocus a;
		float metric = a.BayesMeasure(sample_gray);

		cv::rectangle(sample, cv::Rect(a.m_x, a.m_y, a.delta_w, a.delta_w), cv::Scalar(255, 255, 0));

		char str[200];
		sprintf_s(str, "focus measure: %f", metric);

		cv::putText(sample, str, cv::Point2f(50, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);
		cv::putText(sample, filename.c_str(), cv::Point2f(50, 75), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);

		cv::imshow("original image", sample);
		cv::waitKey(500);

		cvplot::figure("Auto Focus Change").series("BayesDCT").addValue(metric).type(cvplot::DotLine).color(cvplot::Green);
		cvplot::figure("Auto Focus Change").show();
	}

	cv::waitKey(0);

	system("PAUSE");
}
