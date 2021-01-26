#include <iostream>
#include <chrono>

#include "include/FilterExecutor.h"
#include "helper_cuda.h"

int main(int argc, char** argv)
{
	std::string filename("test_images\\cat.png");
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