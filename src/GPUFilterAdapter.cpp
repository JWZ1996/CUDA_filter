#include "../include/GPUFilterAdapter.h"
#include "../include/CudaProcessor.h"

void GPUFilterAdapter::apply(cv::Mat& img)
{
	//CudaProcessor(img).apply();
}

void FFTGPUFilterAdapter::apply(cv::Mat& img)
{
	CudaProcessor(img).apply();
}