#pragma once
#include  "opencv2\opencv.hpp"

struct IGPUFilter
{
	virtual void filter(std::vector<float> &channel) = 0;

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

	void filter(std::vector<float> &channel) override;
};

class FFTGPUFilter : public BaseGPUFilter
{
private:
	// image params
	size_t width;
	size_t height;
	size_t cutoff_frequencies = 0.1;

  void filter(cv::Mat channel);
public:

	FFTGPUFilter(cv::Mat& inputImg);

	void filter(std::vector<float> &channel) override;
};

