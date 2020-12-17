#include <iostream>
#include <string>

#include "include/FilterExecutor.h"
#include "include/cudaPtr.h"




int main()
{
	std::string filename("e:\\studia\\20Z\\RIM\\lena.png");
	cv::Mat input = cv::imread(filename,cv::IMREAD_COLOR);

	//FilterExecutor(FilterExecutorMode::CPU).exec(input);

	cv::imshow(filename, input);

	cv::waitKey(); // unhandled exception

    return 0;
}

