#ifndef __FFT_GPU_COMMON_H__
#define __FFT_GPU_COMMON_H__

#include <vector>

struct Parameters
{
	size_t width, height;
	float cutoff_frequencies;
	bool isLowPass;
};

extern "C"
void filter_CUFFT(float* channel0, float* channel1, float* channel2, const Parameters& params); // main filtering function

#endif