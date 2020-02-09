#include "helpers.h"


class MatContainer
{
	Mat m_mat;
	string m_sMessage = "";
public:
	MatContainer(Mat _mat) : m_mat(_mat)
	{

	}

	MatContainer(Mat _mat, string _message)
	{
		m_mat = _mat;
		m_sMessage = _message;
	}

	MatContainer(const MatContainer& _m) : m_mat(_m.m_mat), m_sMessage(_m.m_sMessage)
	{
		cout << "I'm a copy" << endl;
	}

	~MatContainer()
	{
		cout << "release called for " << m_sMessage << endl;
		m_mat.release();
	}

	char& operator[](unsigned int _i)
	{
		return m_sMessage[_i];
	}

	friend ostream& operator<<(ostream& _oss, const MatContainer& _mat);

};

ostream& operator<<(ostream& _oss, const MatContainer& _mat)
{
	_oss << _mat.m_sMessage;
	return _oss;
}

void printMessage(MatContainer _m)
{
	cout << _m << endl;
}

void printMessageBetter(const MatContainer& _m) // we can't pass temporary R values
{
	cout << _m << endl;
}

void workWithVector(vector<Mat> _v)
{
	_v[0] = _v[0] * 2;
}

void workWithVectorReference(vector<Mat>& _v)
{
	_v[0] = _v[0] * 2;
}

int main(int argc, char** argv)
{
	Mat im1 = imread("rickRights.png"); // original image
	Mat im2 = im1; // shallow copy: data_ptr will have same addresses

	cout << &im1 << " - " << &im2<< endl; // normally, each value will have it's own unique address in the stack

	cout << im1.u->refcount << endl;
	im2 *= 3; // will modify both im1 and im2

	im2.release(); // only removes im2, im1 will stay valid but reference count will decrease
	cout << im1.u->refcount << endl;

	//////////////////////////////////////////////////////////////////////////

	Mat im3 = im1.clone(); // deep copy, completely fresh memory space
	vector<Mat> mats(5, im3); // mat will contain 5 im3, all with the same addresss
	im3 *= 3; // will modify all elements in mats vector
	cout << im3.u->refcount << endl;

	mats[0].release(); // release the first element; refcount expected to be decreased
	cout << im3.u->refcount << endl;

	mats.clear(); // vector clear will the rest of them
	cout << im3.u->refcount << endl;

	//////////////////////////////////////////////////////////////////////////

	Mat im4 = im3(Rect(0, 0, 200, 200)); // still points to same memory with im3, only the data size will be different 
	im4 /= 3;
	cout << im3.u->refcount << endl; // expected to be increased

	im4.release();

	//////////////////////////////////////////////////////////////////////////

	MatContainer m1 = im4;
	MatContainer m2 = m1;

	cout << "////////////////////////////////////////////////////////////////////////// " << endl;

	MatContainer m3(im4, "m3");
	MatContainer m4(m3);

	printMessage(m3); // since printMessage value based function, first copy constructor will be called 
					  // then relese will be called when out of scope
	
	printMessageBetter(m4); // no such thing will happen

	////////////////////////////////////////////////////////////////////////// vector<mat> as a function parameter
	im1 = imread("rickRights.png"); // restore original image
	vector<Mat> mats2(2, im1); // will hold two im1 object with same memory address, im1.clone doesn't change anything
	workWithVector(mats2); 
	workWithVectorReference(mats2); // with reference is also the same

	//////////////////////////////////////////////////////////////////////////
	im1 = imread("rickRights.png"); // restore original image
	vector<Mat> mats3(2, Mat());

	im1.copyTo(mats3[0]);
	im1.copyTo(mats3[1]); // now mats3[0] and mats3[1] different addresses

	waitKey(0);
	
	cout << "I'm done !\n";
	system("PAUSE");
}
