#pragma once


#include  "cudaPtr.h"
#include  "GPUFilters.h"


class CudaProcessor
{
	cv::Mat& inputMat;
	IGPUFilter& gpuFilter;

	std::vector<cv::Mat> inputChannels;
	std::vector<uchar*> gpuChannelsData;
	
	void mergeChannels();
	void splitChannels();
	void channelToGpu(cv::Mat& channel);
	cv::Mat channelToHost(uchar* gpuChannel);

	size_t getImgSize();
	size_t getImgBSize();

public:

	CudaProcessor(cv::Mat& input, IGPUFilter& filter);

	void apply();
};

