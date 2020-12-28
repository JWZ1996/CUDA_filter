#include "FilterAdapters.h"

#include <iostream>
#include <vector>
#include <cmath>

#define _USE_MATH_DEFINES
#define FILTER_ORDER 1 // 1, ..., N; FILTER_ORDER = 2 * FILTER_ORDER + 1, FILTER_ORDERxFILTER_ORDER

//SELECT ONLY ONE TYPE OF FILTER:
//#define SOBEL_FILTER
#define MEDIAN_FILTER
//#define SHARPEN_FILTER
//#define GAUSIAN_FILTER

void CPUFilterAdapter::apply(cv::Mat& img)
{
	//rows = y
	//cols = x
	const int cols = img.cols, rows = img.rows;

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

	img.convertTo(img, CV_32F, 1/255.0f);
	cv::Mat channels[3], merged_photo;
	cv::split(img, channels);

	float sum[3] = { 0.0f, 0.0f, 0.0f };
	for (int img_y = 0; img_y <= rows - kernelSize; img_y++)
	{
		for (int img_x = 0; img_x <= cols - kernelSize; img_x++)
		{
			sum[0] = sum[1] = sum[2] = 0.0f;

			for (int kernel_y = 0; kernel_y < kernel.rows; kernel_y++)
			{
				for (int kernel_x = 0; kernel_x < kernel.cols; kernel_x++)
				{
					for(int i = 0; i < 3; i++)
						sum[i] = sum[i] + kernel.at<float>(kernel_y, kernel_x) * channels[i].at<float>(img_y + kernel_y, img_x + kernel_x);

				}

			}

			for (int i = 0; i < 3; i++)
			{
				if (sum[i] > 1.0f)
					channels[i].at<float>(img_y, img_x) = 1.0f;
				else if (sum[i] < 0.0f)
					channels[i].at<float>(img_y, img_x) = 0.0f;
				else
					channels[i].at<float>(img_y, img_x) = sum[i];
			}			
		}
	}

	std::vector<cv::Mat> channels_vec = {channels[0], channels[1], channels[2]};
	cv::merge(channels_vec, img);
	//cv::imshow("merged", merged_photo);

	//cv::filter2D(img, img, -1, kernel);
}

void GPUFilterAdapter::apply(cv::Mat& img)
{
	ConvGPUFilter filter{ img };

	CudaProcessor(img, filter).apply();
}

void FFTGPUFilterAdapter::apply(cv::Mat& img)
{
	FFTGPUFilter filter{ img };

	CudaProcessor(img, filter).apply();
}