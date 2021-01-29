#include <iostream>
#include <chrono>
#include <numeric>
#include <cmath>

#include "include/FilterExecutor.h"
#include "helper_cuda.h"

void compare(cv::Mat& before, cv::Mat& after)
{
	cv::Mat errMat;
	std::vector<cv::Mat> channelsSplit;
	before.copyTo(errMat);

	float diffSum = 0.0f;

	for (size_t row = 0; row < before.rows; row++)
	{
		for (size_t col = 0; col < before.cols; col++)
		{
			for (size_t channel = 0; channel < before.channels(); channel++)
			{
				float diff = std::abs(errMat.at<cv::Vec3f>(row, col)[channel] - after.at<cv::Vec3f>(row, col)[channel]);
				errMat.at<cv::Vec3f>(row, col)[channel] = diff;
				diffSum += diff;
			}
		}
	}

	cv::imshow("Difference", errMat);
	std::cout << "Error sum: " << diffSum << '\n';
	std::cout << "Error mean: " << ((double)diffSum) / (before.rows * before.cols * 3) << '\n';
}

int main(int argc, char** argv)
{
	std::string filename("test_images\\UVmap.png");
	cv::Mat input = cv::imread(filename, cv::IMREAD_COLOR);
	cv::Mat orig = cv::imread(filename, cv::IMREAD_COLOR);
	cv::imshow(filename, orig);

	auto start = std::chrono::steady_clock::now();
	FilterExecutor(FilterExecutorMode::GPU_FFT).exec(input);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Overall time " << elapsed_seconds.count() << "s time\n";

	cv::imshow(filename, input);
	cv::waitKey(); // unhandled exception

	return 0;
}