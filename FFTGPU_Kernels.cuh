# pragma once
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

typedef float2 fComplex;

__constant__  float2 dZero;

__global__ void lowPassFilter(
	fComplex *dData,
	int fftH,
	int fftW,
	int imageW,
	int imageH,
	const size_t sqradius
);

__global__ void highPassFilter(
	fComplex *dData,
	int fftH,
	int fftW,
	int imageW,
	int imageH,
	const size_t sqradius
);

__device__ void cutFrequencies(
	fComplex *dData,
	int fftH,
	int fftW,
	int imageW,
	int imageH,
	const size_t sqradius,
	bool(*func)(int, int, size_t)

);

__global__ void fftShift(
	fComplex *dData,
	int fftH,
	int fftW
);

// for cufft callback

struct ImageInfo 
{
	size_t width;
	size_t height;
	size_t centerX;
	size_t centerY;
};


__device__ void cufftCallbackShift(
	void *dataOut,
	size_t offset,
	cufftComplex  element,
	void *callerInfo,
	void *sharedPointer
);


//#include "kernels.cu"