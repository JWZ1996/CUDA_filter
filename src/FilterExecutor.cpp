#include "../include/FilterExecutor.h"

void FilterExecutor::exec(cv::Mat& imgRef)
{
	filterpool.at(mode)->apply(imgRef);
}