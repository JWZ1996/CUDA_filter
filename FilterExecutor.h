#pragma once
#include <array>
#include <memory>

#include "FilterAdapters.h"


typedef std::array<std::unique_ptr<IFilterAdapter>,3u> FilterPool;

enum FilterExecutorMode
{
	CPU     = 0u,
	GPU     = 1u,
	GPU_FFT = 2u
};

class FilterExecutor
{
	cv::Mat& transformedImg;


	FilterPool filterpool
	{
		std::make_unique<CPUFilterAdapter>(),
		std::make_unique<GPUFilterAdapter>(),
		std::make_unique<FFTGPUFilterAdapter>(),
	};
public:
	FilterExecutor(cv::Mat& imgRef);


	void exec(FilterExecutorMode mode);
};

