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

// DODANIE OBRAMOWANIA DO ZDJEC
	int x = FILTER_ORDER,
		y = FILTER_ORDER;

	if (cols < rows)
	{
		int diff = std::abs(cols - rows);

		if (diff % 2 == 0) {
			cv::copyMakeBorder(img, img, 0, 0, diff / 2, diff / 2, cv::BORDER_CONSTANT);
			x += (diff / 2);
		}
		else {
			int split = diff / 2;

			cv::copyMakeBorder(img, img, 0, 0, split + 1, split, cv::BORDER_CONSTANT);
			x += split;
		}

	}
	else if (cols > rows) {
		int diff = std::abs(cols - rows);

		if (diff % 2 == 0) {
			cv::copyMakeBorder(img, img, diff / 2, diff / 2, 0, 0, cv::BORDER_CONSTANT);
			y += (diff / 2);
		}
		else {
			int split = diff / 2;

			cv::copyMakeBorder(img, img, split + 1, split, 0, 0, cv::BORDER_CONSTANT);
			y += split;
		}
	}

	cv::copyMakeBorder(img, img, FILTER_ORDER, FILTER_ORDER, FILTER_ORDER, FILTER_ORDER, cv::BORDER_CONSTANT);
	//--------------------


	img.convertTo(img, CV_32F, 1/255.0f);
	cv::Mat channels[3], merged_photo;
	cv::split(img, channels);

	/*float sum[3] = { 0.0f, 0.0f, 0.0f };

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
	}*/

	for (int i = 0; i < 3; i++)
		cv::filter2D(channels[i], channels[i], -1, kernel);

	std::vector<cv::Mat> channels_vec = {channels[0], channels[1], channels[2]};
	cv::merge(channels_vec, img);
}

void GPUFilterAdapter::apply(cv::Mat& img)
{
	const int 
		cols = img.cols,
		rows = img.rows;

	int x = FILTER_ORDER,
		y = FILTER_ORDER;

	if (cols < rows)
	{
		int diff = std::abs(cols - rows);

		if (diff % 2 == 0) {
			cv::copyMakeBorder(img, img, 0, 0, diff/2, diff/2, cv::BORDER_CONSTANT);
			x += (diff / 2);
		}
		else {
			int split = diff / 2;

			cv::copyMakeBorder(img, img, 0, 0, split + 1, split, cv::BORDER_CONSTANT);
			x += split;
		}

	}
	else if (cols > rows) {
		int diff = std::abs(cols - rows);

		if (diff % 2 == 0) {
			cv::copyMakeBorder(img, img, diff / 2, diff / 2, 0, 0, cv::BORDER_CONSTANT);
			y += (diff / 2);
		}
		else {
			int split = diff / 2;

			cv::copyMakeBorder(img, img, split + 1, split, 0, 0, cv::BORDER_CONSTANT);
			y += split;
		}
	}
	
	cv::copyMakeBorder(img, img, FILTER_ORDER, FILTER_ORDER, FILTER_ORDER, FILTER_ORDER, cv::BORDER_CONSTANT);
	ConvGPUFilter filter{ img };
	CudaProcessor(img, filter).apply();

#ifdef CONVFILTER3
	auto rec = cv::Rect(x, y, cols, rows);
#endif	

#ifdef CONVFILTER2
	auto rec = cv::Rect(FILTER_ORDER, FILTER_ORDER, cols, rows);
#endif	

#ifdef CONVFILTER1
	auto rec = cv::Rect(FILTER_ORDER, FILTER_ORDER, cols, rows);
#endif
	//img = img(rec);
}

void FFTGPUFilterAdapter::apply(cv::Mat& img)
{
	FFTGPUFilter filter(img);

	CudaProcessor(img, filter).apply();
}