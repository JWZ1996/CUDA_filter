#pragma once
#include  "opencv2\opencv.hpp"

struct IGPUFilter
{
private:
	virtual void filter(std::vector<float> &channel) = 0;
public:
	virtual void filterWrapper(std::vector<std::vector<float>>&) = 0;
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
private:
	void filter(std::vector<float> &channel) override;
public:
	ConvGPUFilter(cv::Mat& inputImg);

	void filterWrapper(std::vector<std::vector<float>>&) override;
};

class FFTGPUFilter : public BaseGPUFilter
{
private:
	// image params
	size_t width;
	size_t height;
	size_t cutoff_frequencies = 0.1;

	void filter(std::vector<float> &channel) override;
public:

	FFTGPUFilter(cv::Mat& inputImg);

	void filterWrapper(std::vector<std::vector<float>>&) override;
};

