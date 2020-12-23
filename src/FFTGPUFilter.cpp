#include "../include/GPUFilters.h"
// cuh kernel

FFTGPUFilter::FFTGPUFilter(cv::Mat& inputImg) : BaseGPUFilter(inputImg)
{
	// instatiate kernel params
	kWidth = inputImg.cols / 10;
	kHeight = inputImg.rows / 10;

	
}


void FFTGPUFilter::filter(uchar* channel) // only 1-channel
{
	for (int i = 0; i < kWidth * kHeight; i++)
		channel[i] = 255;
}