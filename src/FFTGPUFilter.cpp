#include "../include/GPUFilters.h"


#include "../FFTGPU_Common.h"


FFTGPUFilter::FFTGPUFilter(cv::Mat& inputImg) : BaseGPUFilter(inputImg) , width(inputImg.cols), height(inputImg.rows) { }

void FFTGPUFilter::filter(std::vector<float>& channel) // only 1-channel
{
	//filter_CUFFT(&channel[0], width, height, 0.1f, false);
}

void FFTGPUFilter::filterWrapper(std::vector<std::vector<float>>& channels)
{
	Parameters params;
	params.width = width;
	params.height = height;
	params.cutoff_frequencies = 0.1f;
	params.isLowPass = false;
	filter_CUFFT(&channels[0].at(0), &channels[1].at(0), &channels[2].at(0), params);
	//std::for_each(channels.begin(), channels.end(), [&](auto& channel) { this->filter(channel); });
}