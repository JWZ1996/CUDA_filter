#include <iostream>
#include "FilterExecutor.h"
#include "cudaPtr.h"



int main()
{
	std::string filename{ "C:\\Users\\J A N E C K\\Desktop\\test.jpg" };

	cv::Mat input = cv::imread(filename,cv::IMREAD_COLOR);

	FilterExecutor(input).exec(FilterExecutorMode::GPU);

	cv::imshow(filename, input);

	cv::waitKey();

    return 0;
}

