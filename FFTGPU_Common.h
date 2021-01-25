#ifndef __FFT_GPU_COMMON_H__
#define __FFT_GPU_COMMON_H__

extern "C"
void filter_CUFFT(float* hChannel, size_t hWidth, size_t hHeight, const float cutoff_frequencies, const bool isLowPass); // main filtering function

#endif