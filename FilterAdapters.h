#pragma once
#include  "opencv2\opencv.hpp"

// Interface for a filtering system
struct IFilterAdapter
{
	virtual void apply(cv::Mat& img) = 0;
};


// Filter-adapters implementations:

struct CPUFilterAdapter : public IFilterAdapter
{
	void apply(cv::Mat& img) override;
	~CPUFilterAdapter() = default;
};
struct GPUFilterAdapter : public IFilterAdapter
{
	void apply(cv::Mat& img) override;
	~GPUFilterAdapter() = default;
};
struct FFTGPUFilterAdapter : public IFilterAdapter
{
	void apply(cv::Mat& img) override;
	~FFTGPUFilterAdapter() = default;
};
