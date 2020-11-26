#include "GPUFilterAdapter.h"
#include "CudaProcessor.h"
#include "cudaPtr.h"

void GPUFilterAdapter::apply(cv::Mat& img)
{
	CudaProcessor(img).apply();
}

void FFTGPUFilterAdapter::apply(cv::Mat& img)
{
	CudaProcessor(img).apply();
}
