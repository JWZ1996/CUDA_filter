#include "../include/CudaProcessor.h"
#include "helper_cuda.h"

#include <algorithm>
CudaProcessor::CudaProcessor(cv::Mat& input, IGPUFilter& filter) : gpuFilter{ filter }, inputMat(input)
{
	inputMat.convertTo(inputMat, CV_32F, 1 / 255.0f);
	splitChannels();
}

void CudaProcessor::splitChannels()
{
	cv::split(inputMat, inputChannels);

	std::for_each(std::begin(inputChannels), std::end(inputChannels), [this](auto& channel)
	{
		channelToGpu(channel);
	});

}
void CudaProcessor::channelToGpu(cv::Mat& channel)
{
	std::vector<float> data;

	if (channel.isContinuous())
	{
		data.assign((float*)channel.data, (float*)channel.data + channel.total() * channel.channels());
	}
	else
	{
		for (int i = 0; i < channel.rows; ++i)
		{
			data.insert(data.end(), channel.ptr<float>(i), channel.ptr<float>(i) + channel.cols * channel.channels());
		}
	}

	gpuChannelsData.emplace_back(data);
}

void CudaProcessor::apply()
{

	std::for_each(std::begin(gpuChannelsData), std::end(gpuChannelsData), [this](auto& channel)
	{

		// Kanał w osobnym strumieniu
		// Wersja batchowana
		gpuFilter.filter(channel);
		// no resource deallocation?
	});

	for (int k = 0; k < inputChannels.size(); k++)
	{
		for (int i = 0; i < inputMat.rows; i++)
		{
			for (int j = 0; j < inputMat.cols; j++)
			{
				inputChannels[k].at<float>(i, j) = gpuChannelsData[k][i * inputMat.rows + j];
			}
		}
	}

	cv::merge(inputChannels, inputMat);
}