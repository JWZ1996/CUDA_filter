// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <math.h>

#include "FFTGPU_Common.h"
#include "FFTGPU_Kernels.cuh"



//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}


// fit to nearest mult of 512
int snapTransformSize(int dataSize)
{
	int hiBit;
	unsigned int lowPOT, hiPOT;

	dataSize = iAlignUp(dataSize, 16);

	for (hiBit = 31; hiBit >= 0; hiBit--)
		if (dataSize & (1U << hiBit))
		{
			break;
		}

	lowPOT = 1U << hiBit;

	if (lowPOT == (unsigned int)dataSize)
	{
		return dataSize;
	}

	hiPOT = 1U << (hiBit + 1);

	if (hiPOT <= 1024)
	{
		return hiPOT;
	}
	else
	{
		return iAlignUp(dataSize, 512);
	}
}

void padData(float* unpadded, fComplex* padded, const size_t fftH, const size_t fftW, const size_t hW, const size_t hH, const bool forward=true) {
	const float norm = 1.0f / (fftW * fftH);
	for(int x = 0; x < hW; x++)
		for (int y = 0; y < hH; y++) {
			if (forward) {
				padded[fftW * y + x].x = unpadded[hW * y + x];
				padded[fftW * y + x].y = 0.0f;
			}	
			else {
				unpadded[hW * y + x] = padded[fftW * y + x].x * norm;
			}
				
		}
}


size_t square_radius(const size_t width, const size_t height, const float cutoff_frequencies)
{
	const size_t mind = width > height ? height : width;
	const size_t radius = static_cast<size_t>((mind / 2) * cutoff_frequencies);
	return radius * radius;
}



void filter_CUFFT(float* hChannel, size_t hWidth, size_t hHeight, const float cutoff_frequencies, const bool isLowPass)
{	
	// vars declaration, data-related
	fComplex * dChannelPadded, *hPadded;
	// vars declaration, fft-related

	//		create fft plan
	cufftHandle
		fftPlanFwd, 
		fftPlanInv;

	//		spectrum
	fComplex *dDataSpectrum;


	size_t fftWidth = snapTransformSize(hWidth);
	size_t fftHeight = snapTransformSize(hHeight);
	
	// calculate transformation radius
	const size_t sqradius = square_radius(fftWidth, fftHeight, cutoff_frequencies);
	// memory allocalion
	hPadded = (fComplex *)malloc(fftWidth * fftHeight * sizeof(fComplex));
	memset(hPadded, 0, fftWidth * fftHeight * sizeof(fComplex));
	// gpu memory allocation
	checkCudaErrors(cudaMalloc((void **)&dChannelPadded, fftWidth * fftHeight * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&dDataSpectrum, fftHeight * fftWidth * sizeof(fComplex)));

	// memory upload
	const float2 hZero = make_float2(0.0f, 0.0f);
	cudaMemcpyToSymbol(&dZero, &hZero, sizeof(float2));
	printf("s %d %d", hWidth, hHeight, sqradius);
	
	padData(hChannel, hPadded, fftWidth, fftHeight, hWidth, hHeight);
	checkCudaErrors(cudaMemcpy(dChannelPadded, hPadded, fftWidth * fftHeight * sizeof(cufftComplex), cudaMemcpyHostToDevice));
	// *************************
	// *** transformations *****
	// *************************

	// create fft plans
	checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftWidth, fftHeight, CUFFT_C2C));
	checkCudaErrors(cufftPlan2d(&fftPlanInv, fftWidth, fftHeight, CUFFT_C2C));

	// fft convolution
	checkCudaErrors(cudaDeviceSynchronize());
	
	
	cudaEvent_t start, stop; // pomiar czasu wykonania jądra
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	checkCudaErrors(cufftExecC2C(fftPlanFwd, (cufftComplex *)dChannelPadded, (cufftComplex *)dDataSpectrum, CUFFT_FORWARD));



	dim3 block(32, 32);
	dim3 grid(iDivUp(fftWidth, block.x), iDivUp(fftHeight, block.y));
	
	

	/* ###### KERNELS ###### */
	// cut frequencies kernel
	fftShift << <grid, block >> > (dDataSpectrum, fftWidth, fftHeight);
	if(isLowPass)
		lowPassFilter <<<grid, block >>> (
			dDataSpectrum,
			fftHeight,
			fftWidth,
			hHeight,
			hWidth,
			sqradius
		);
	else
		highPassFilter << <grid, block >> > (
			dDataSpectrum,
			fftHeight,
			fftWidth,
			hHeight,
			hWidth,
			sqradius
			);

	fftShift << <grid, block >> > (dDataSpectrum, fftWidth, fftHeight);
	/* ######### END KERNELS ####### */

	


	//// inverse transformation
	checkCudaErrors(cufftExecC2C(fftPlanInv, (cufftComplex *)dDataSpectrum, (cufftComplex *)dChannelPadded, CUFFT_INVERSE));

	
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	printf("GPU (kernel) time = %.3f ms\n", elapsedTime);

	checkCudaErrors(cudaDeviceSynchronize());
	
	checkCudaErrors(cudaMemcpy(hPadded, dChannelPadded, fftWidth  * fftHeight * sizeof(fComplex), cudaMemcpyDeviceToHost));
	
	padData(hChannel, hPadded, fftWidth, fftHeight, hWidth, hHeight, false);
	
	// normalize
	
	// free resources
	checkCudaErrors(cufftDestroy(fftPlanInv));
	checkCudaErrors(cufftDestroy(fftPlanFwd));

	checkCudaErrors(cudaFree(dDataSpectrum));

	checkCudaErrors(cudaFree(dChannelPadded));
	free(hPadded);
}