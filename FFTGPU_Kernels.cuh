# pragma once

typedef float2 fComplex;
#include "kernels.cu"
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
