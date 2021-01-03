#pragma once

#include "BaseFilterAdapter.h"
#include "CudaProcessor.h"

class GPUFilterAdapter : public BaseFilterAdapter
{
	void apply(cv::Mat& img) override;
};

class FFTGPUFilterAdapter : public BaseFilterAdapter
{
	void apply(cv::Mat& img) override;
};
