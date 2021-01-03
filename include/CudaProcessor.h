#pragma once

#include  "GPUFilters.h"

class CudaProcessor
{
	cv::Mat& inputMat;
	IGPUFilter& gpuFilter;

	std::vector<cv::Mat> inputChannels;
	std::vector<std::vector<float>> gpuChannelsData;

	void splitChannels();
	void channelToGpu(cv::Mat& channel);

public:

	CudaProcessor(cv::Mat& input, IGPUFilter& filter);

	void apply();
};
