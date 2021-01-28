#include <iostream>
#include <iterator>
#include <numeric>

#include "../include/GPUFilters.h"
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include "../include/KernelGenerator.h"
#include "../include/CompilationFlags.h"

const int TILE_SIZE = 16;
__constant__ float KERNEL[(2 * FILTER_ORDER + 1) * (2 * FILTER_ORDER + 1)];

// WERSJA CONVFILTER3 - ze stala maska
__global__ void ConvFilter3(float* input, float* output, int width, int height) {

	float sum = 0.0f;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ((row < (height - (KERNEL_SIZE - FILTER_ORDER)) && row >= 0) && (col < (width - (KERNEL_SIZE - FILTER_ORDER)) && col >= 0)) {

#pragma unroll
		for (int i = 0; i < KERNEL_SIZE; i++) {
#pragma unroll
			for (int j = 0; j < KERNEL_SIZE; j++) {

				sum += input[(row + i) * width + col + j] * KERNEL[i * KERNEL_SIZE + j];
			}

		}
		output[row * width + col] = sum;
	

// WERSJA CONVFILTER2 - ze stala maska
__global__ void ConvFilter2(float* input, float* output, int width, int height, int kernelSize) {

	float sum = 0.0f;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int maskRowsRadius = kernelSize >> 1;
	int maskColsRadius = kernelSize >> 1;

	if (row < height && col < width) {
		sum = 0.0f;
		int startRow = row - maskRowsRadius;
		int startCol = col - maskColsRadius;

		for (int i = 0; i < kernelSize; i++) {

			for (int j = 0; j < kernelSize; j++) {

				int currentRow = startRow + i;
				int currentCol = startCol + j;

				if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width)
				{

					sum += input[currentRow * width + currentCol] * KERNEL[i * kernelSize + j];
				}
			}
		}
		output[row * width + col] = sum;
	}
}

// WERSJA CONVFILTER1 - podstawowa
__global__ void ConvFilter1(float* input,  float*  kernel,
	float* output, int width, int height, int kernelSize) {


	float sum = 0.0f;
	int col = threadIdx.x + blockIdx.x * blockDim.x;   
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int maskRowsRadius = kernelSize >> 1;
	int maskColsRadius = kernelSize >> 1;

		if (row < height && col < width) {
			sum = 0.0f;
			int startRow = row - maskRowsRadius;  
			int startCol = col - maskColsRadius;  

			for (int i = 0; i < kernelSize; i++) {

				for (int j = 0; j < kernelSize; j++) {

					int currentRow = startRow + i;
					int currentCol = startCol + j;

					if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) 
					{

						sum += input[currentRow * width + currentCol] * kernel[i * kernelSize + j];
					}
				}
			}
			output[row * width + col] = sum;
		}
}

ConvGPUFilter::ConvGPUFilter(cv::Mat& inputImg) : BaseGPUFilter(inputImg) {}

void ConvGPUFilter::filter(std::vector<float>& channel)
{
	//rows = y
	//cols = x
	const int cols = inputMat.cols, rows = inputMat.rows;
	const int kernelSize = 2 * FILTER_ORDER + 1;

	float *kernelInput = new float[kernelSize * kernelSize];
	float *photoInput = &channel[0];
	float *photoOutput = new float[cols * rows];

#ifdef MEDIAN_KERNEL
		KernelGenerator::generateMedianKernel(kernelInput, kernelSize);
#endif

#ifdef GAUSIAN_KERNEL
		KernelGenerator::generateGaussianKernel(kernelInput, kernelSize, SIGMA);
#endif

#ifdef PREWITT_HORIZONTAL_KERNEL
		KernelGenerator::returnPrewittHorizontalKernel(kernelInput);
#endif	

#ifdef PREWITT_VERTICAL_KERNEL
		KernelGenerator::returnPrewittVerticalKernel(kernelInput);
#endif

#ifdef SOBEL_HORIZONTAL_KERNEL
		KernelGenerator::returnSobelHorizontalKernel(kernelInput);
#endif	

#ifdef SOBEL_VERTICAL_KERNEL
		KernelGenerator::returnSobelVerticalKernel(kernelInput);
#endif

	checkCudaErrors(cudaSetDevice(0));

	float *devInputPhoto,  
		  *devOutputPhoto,
		  *devKernel;
		
	checkCudaErrors(cudaMalloc((void**)&devInputPhoto, cols * rows  * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&devKernel, kernelSize * kernelSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&devOutputPhoto, cols * rows * sizeof(float)));

	checkCudaErrors(cudaMemcpy(devInputPhoto, photoInput, cols * rows * sizeof(float), cudaMemcpyHostToDevice));

	dim3 dimGrid(ceil((float)cols / TILE_SIZE),
		ceil((float)rows / TILE_SIZE));
	dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

#ifdef CONVFILTER3
	checkCudaErrors(cudaMemcpyToSymbol(KERNEL, kernelInput, kernelSize * kernelSize * sizeof(float)));
	ConvFilter3 <<< dimGrid, dimBlock >>> (devInputPhoto, devOutputPhoto, cols, rows);
#endif	

#ifdef CONVFILTER2
	checkCudaErrors(cudaMemcpyToSymbol(KERNEL, kernelInput, kernelSize * kernelSize * sizeof(float)));
	ConvFilter2 <<< dimGrid, dimBlock >>> (devInputPhoto, devOutputPhoto, cols, rows, kernelSize);
#endif	

#ifdef CONVFILTER1
	checkCudaErrors(cudaMemcpy(devKernel, kernelInput, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));
	ConvFilter1 <<< dimGrid, dimBlock >>> (devInputPhoto, devKernel, devOutputPhoto, cols, rows, kernelSize);
#endif

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(photoOutput, devOutputPhoto, cols * rows * sizeof(float), cudaMemcpyDeviceToHost));

	//przeniesienie danych z tablicy do vectora
	std::memcpy(&channel[0], photoOutput, cols * rows * sizeof(float));

	cudaFree(devInputPhoto);
	cudaFree(devOutputPhoto);
	cudaFree(devKernel);

	photoInput = 0;
	delete kernelInput;
	delete photoInput;
	delete photoOutput;
}