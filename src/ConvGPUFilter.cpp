#include "GPUFilters.h"

#define _USE_MATH_DEFINES
#define FILTER_ORDER 1 // 1, ..., N; FILTER_ORDER = 2 * FILTER_ORDER + 1, FILTER_ORDERxFILTER_ORDER

//SELECT ONLY ONE TYPE OF FILTER:
//#define SOBEL_FILTER
#define MEDIAN_FILTER
//#define SHARPEN_FILTER
//#define GAUSIAN_FILTER

ConvGPUFilter::ConvGPUFilter(cv::Mat& inputImg) : BaseGPUFilter(inputImg)
{

}

void ConvGPUFilter::filter(uchar* channel)
{

	//rows = y
	//cols = x
	const int cols = this->inputMat.cols, rows = this->inputMat.rows;

	int kernelSize = 2 * FILTER_ORDER + 1;

	cv::Mat  kernel{ cv::Mat::ones(kernelSize, kernelSize, CV_32F) };

#ifdef MEDIAN_FILTER
	kernel = kernel / (float)(kernelSize * kernelSize);
#endif

#ifdef SOBEL_FILTER
	cv::Mat  kernel{ cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize) };
#endif

#ifdef GAUSIAN_FILTER
	for (int i = 0; i < kernelSize; i++)
	{
		for (int j = 0; j < kernelSize; j++)
		{
			kernel.at<float>(i, j) = (1 / (2 * 3.14 * 0.5)) * expf(-(i * i + j * j) / (2 * 0.5 * 0.5));
		}
	}
#endif

	uchar sum = 0u;
	for (int img_y = 0; img_y <= rows - kernelSize; img_y++)
	{
		for (int img_x = 0; img_x <= cols - kernelSize; img_x++)
		{
			sum = 0u;

			for (int kernel_y = 0; kernel_y < kernel.rows; kernel_y++)
			{
				for (int kernel_x = 0; kernel_x < kernel.cols; kernel_x++)
				{
						//sum = sum + kernel.at<float>(kernel_y, kernel_x) * channel(img_y + kernel_y, img_x + kernel_x);

				}

			}

			/*if (sum > 255u)
				channel(img_y, img_x) = 255u;
			else if (sum < 0u)
				channel(img_y, img_x) = 0u;
			else
				channel(img_y, img_x)= sum;*/
			
		}
	}
}