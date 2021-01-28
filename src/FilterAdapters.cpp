#include "../include/FilterAdapters.h"
#include "../include/CompilationFlags.h"
#include "../include/KernelGenerator.h"
#include <vector>

void CPUFilterAdapter::apply(cv::Mat& img)
{
	//rows = y
	//cols = x
	const int cols = img.cols, rows = img.rows;

	int kernelSize = 2 * FILTER_ORDER + 1;

	cv::Mat  kernel{ cv::Mat::ones(kernelSize, kernelSize, CV_32F) };
	
	float* kernelInput = new float[kernelSize * kernelSize];


#ifdef MEDIAN_KERNEL
	KernelGenerator::generateMedianKernel(kernelInput, kernelSize);
#endif

#ifdef GAUSIAN_KERNEL
	KernelGenerator::generateGaussianKernel(kernelInput, kernelSize, SIGMA);
#endif

#ifdef PREWITT_HORIZONTAL_KERNEL
	KernelGenerator::returnPrewittHorizontalKernel(kernelInput);
#endif	

#ifdef PREWITT_VERTICAL_KERNEL
	KernelGenerator::returnPrewittVerticalKernel(kernelInput);
#endif

#ifdef SOBEL_HORIZONTAL_KERNEL
	KernelGenerator::returnSobelHorizontalKernel(kernelInput);
#endif	

#ifdef SOBEL_VERTICAL_KERNEL
	KernelGenerator::returnSobelVerticalKernel(kernelInput);
#endif

	for (int i = 0; i < kernelSize; i++)
		for (int j = 0; j < kernelSize; j++)
			kernel.at<float>(i, j) = kernelInput[i * kernelSize + j];

	int x = 0,
		y = 0;
	cv::copyMakeBorder(img, img, 0, 2 * FILTER_ORDER, 0, 2 * FILTER_ORDER, cv::BORDER_CONSTANT);

	img.convertTo(img, CV_32F, 1/255.0f);
	cv::Mat channels[3], merged_photo;
	cv::split(img, channels);

	for (int i = 0; i < 3; i++)
		cv::filter2D(channels[i], channels[i], -1, kernel);

	std::vector<cv::Mat> channels_vec = {channels[0], channels[1], channels[2]};
	cv::merge(channels_vec, img);
	auto rec = cv::Rect(x, y, cols, rows);
	img = img(rec);
}

void GPUFilterAdapter::apply(cv::Mat& img)
{
	const int 
		cols = img.cols,
		rows = img.rows;

	int x = 0,
		y = 0;
	
	cv::copyMakeBorder(img, img, 0, 2 * FILTER_ORDER, 0, 2 * FILTER_ORDER, cv::BORDER_CONSTANT);
	ConvGPUFilter filter{ img };
	CudaProcessor(img, filter).apply();

	auto rec = cv::Rect(x, y, cols, rows);
	img = img(rec);
}

void FFTGPUFilterAdapter::apply(cv::Mat& img)
{
	FFTGPUFilter filter(img);

	CudaProcessor(img, filter).apply();
}