#pragma once

#include  "opencv2\opencv.hpp"

class BaseFilterAdapter
{
	virtual void apply(cv::Mat& img) = 0;
};