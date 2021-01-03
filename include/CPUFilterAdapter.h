#include "BaseFilterAdapter.h"

class CPUFilterAdapter : public BaseFilterAdapter
{
	void apply(cv::Mat& img) override;
};