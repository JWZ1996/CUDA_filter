#pragma once
#include  "opencv2\opencv.hpp"

struct IGPUFilter
{
	virtual void filter(uchar* channel) = 0;

};

class BaseGPUFilter : public IGPUFilter
{
protected:
	cv::Mat& inputMat;
public:
	BaseGPUFilter(cv::Mat& inputImg) : inputMat{ inputImg } {};
};

class ConvGPUFilter : public BaseGPUFilter
{

public:
	ConvGPUFilter(cv::Mat& inputImg);

	void filter(uchar* channel) override;
};

class FFTGPUFilter : public BaseGPUFilter
{

public:
	FFTGPUFilter(cv::Mat& inputImg);

	void filter(uchar* channel) override;
};