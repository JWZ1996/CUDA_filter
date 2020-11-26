#include "FilterExecutor.h"

FilterExecutor::FilterExecutor(cv::Mat& imgRef) : transformedImg { imgRef }
{

}

void FilterExecutor::exec(FilterExecutorMode mode)
{
	filterpool.at(mode)->apply(this->transformedImg);
}
