// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <math.h>

#include "FFTGPU_Common.h"
#include "kernels.cu"



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
	for(int x = 0; x < hW; x++)
		for (int y = 0; y < hH; y++) {
			if (forward) { // TODO: ugly {
				padded[fftW * y + x].x = unpadded[hW * y + x];
				padded[fftW * y + x].y = 0.0f;
			}	
			else {
			unpadded[hW * y + x] = padded[fftW * y + x].x;
			}
				
		}
}


size_t square_radius(const size_t width, const size_t height, const float cutoff_frequencies)
{
	const size_t mind = width > height ? height : width;
	const size_t radius = static_cast<size_t>((mind / 2) * cutoff_frequencies);
	return radius * radius;
}



void filter_CUFFT(float* hChannel, size_t hWidth, size_t hHeight, const float cutoff_frequencies)
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
	checkCudaErrors(cufftExecC2C(fftPlanFwd, (cufftComplex *)dChannelPadded, (cufftComplex *)dDataSpectrum, CUFFT_FORWARD));
	dim3 block(32, 32);
	dim3 grid(iDivUp(fftWidth, block.x), iDivUp(fftHeight, block.y));
	
	// cut frequencies kernel
	
	fftShift << <grid, block >> > (dDataSpectrum, fftWidth, fftHeight);

	//cutFrequencies <<<grid, block >>> (
	//	dDataSpectrum,
	//	fftHeight,
	//	fftWidth,
	//	hHeight,
	//	hWidth,
	//	sqradius
	//);

	fftShift << <grid, block >> > (dDataSpectrum, fftWidth, fftHeight);


	//// inverse transformation
	checkCudaErrors(cufftExecC2C(fftPlanInv, (cufftComplex *)dDataSpectrum, (cufftComplex *)dChannelPadded, CUFFT_INVERSE));

	checkCudaErrors(cudaDeviceSynchronize());
	
	printf("...reading back GPU convolution results\n");
	
	checkCudaErrors(cudaMemcpy(hPadded, dChannelPadded, fftWidth  * fftHeight * sizeof(fComplex), cudaMemcpyDeviceToHost));
	printf("s %f %f", hPadded[0], hPadded[1]);
	padData(hChannel, hPadded, fftWidth, fftHeight, hWidth, hHeight, false);
	
	// normalize
	printf("s %f %f", hChannel[0], hChannel[1]);
	const float norm = 1.0f / (hWidth * hHeight);
	for (int j = 0; j < hWidth * hHeight; j++)
		hChannel[j] = hChannel[j] * norm;
	printf("s %f %f", hChannel[0], hChannel[1]);
	// free resources
	checkCudaErrors(cufftDestroy(fftPlanInv));
	checkCudaErrors(cufftDestroy(fftPlanFwd));

	checkCudaErrors(cudaFree(dDataSpectrum));

	checkCudaErrors(cudaFree(dChannelPadded));
	free(hPadded);
}