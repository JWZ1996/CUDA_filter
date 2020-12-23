#include <iostream>
#include <string>
#include <chrono>

#include "FilterExecutor.h"
#include "cudaPtr.h"

int main()
{
	std::string filename("C:\\lena_std.tif");
	cv::Mat input = cv::imread(filename, cv::IMREAD_COLOR);

	auto start = std::chrono::steady_clock::now();
	FilterExecutor(FilterExecutorMode::CPU).exec(input);
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "The filtration took: " << elapsed_seconds.count() << "s time\n";

	cv::imshow(filename, input);
	cv::waitKey(); // unhandled exception

	return 0;
}