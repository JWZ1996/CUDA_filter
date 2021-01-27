#include "../include/KernelGenerator.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>

void KernelGenerator::generateMedianKernel(float* array, int kernelSize)
{
	float value = 1.0f / (float)kernelSize / (float)kernelSize;
	for (int i = 0; i < kernelSize * kernelSize; i++)
		array[i] = value;
}

void KernelGenerator::generateGaussianKernel(float* array, int kernelSize, float sigma)
{
	for (int i = 0; i < kernelSize; i++)
	{
		for (int j = 0; j < kernelSize; j++)
		{
			array[i * kernelSize + j] = (1 / (2 * M_PI * sigma)) * expf(-(i * i + j * j) / (2 * sigma * sigma));
		}
	}
}

void KernelGenerator::returnPrewittHorizontalKernel(float* array)
{
	array[0] = -1;
	array[1] = -1;
	array[2] = -1;
	array[3] = 0;
	array[4] = 0;
	array[5] = 0;
	array[6] = 1;
	array[7] = 1;
	array[8] = 1;
}

void KernelGenerator::returnPrewittVerticalKernel(float* array)
{
	array[0] = -1;
	array[1] = 0;
	array[2] = 1;
	array[3] = -1;
	array[4] = 0;
	array[5] = 1;
	array[6] = -1;
	array[7] = 0;
	array[8] = 1;
}

void KernelGenerator::returnSobelHorizontalKernel(float* array)
{
	array[0] = -1;
	array[1] = -2;
	array[2] = -1;
	array[3] = 0;
	array[4] = 0;
	array[5] = 0;
	array[6] = 1;
	array[7] = 2;
	array[8] = 1;
}

void KernelGenerator::returnSobelVerticalKernel(float* array)
{
	array[0] = -1;
	array[1] = 0;
	array[2] = 1;
	array[3] = -2;
	array[4] = 0;
	array[5] = 2;
	array[6] = -1;
	array[7] = 0;
	array[8] = 1;
}