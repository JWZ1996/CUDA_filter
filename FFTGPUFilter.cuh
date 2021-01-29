

#include <math.h>
#include <thread>

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

void padData(float* unpadded, fComplex* padded, const size_t fftW, const size_t fftH, const size_t hW, const size_t hH, const bool forward=true) {
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

__device__ void cufftCallbackShift(
	void *dataOut,
	size_t offset,
	cufftComplex  element,
	void *callerInfo,
	void *sharedPointer
)
{
	ImageInfo* i = (ImageInfo*)callerInfo;
	cufftComplex* out = (cufftComplex*)dataOut;


	const int width = i->width;
	const int centerX = i->centerX;
	const int centerY = i->centerY;

	const int x = offset % width;
	const int y = offset / width;

	const int ind = y * width + x;
	int shiftX, shiftY;



	// 1q, 2q
	if (y < centerY) {
		// 1q to 4q
		if (x < centerX) {
			shiftX = x + centerX;
			shiftY = y + centerY;
		}
		//  2q to 3q
		else {
			shiftX = x - centerX;
			shiftY = y + centerY;
		}

	}
	// 3q, 4q
	else {
		// 3q to 2q
		if (x < centerX) {
			shiftX = x + centerX;
			shiftY = y - centerY;
		}
		// 4q to 1q
		else {
			shiftX = x - centerX;
			shiftY = y - centerY;
		}
	}

	//const int indShift = shiftY * imageW + shiftX;
	const int indShift = width * shiftY + shiftX;

	out[ind] = out[indShift];

}

__device__
cufftCallbackStoreC d_cufftshiftCallback = cufftCallbackShift;

struct ThreadInfo {
	float* hChannel;
	fComplex* hPadded;
	size_t fftWidth, fftHeight, hWidth, hHeight;
	bool forward;
	//padData(channel0, hPadded0, fftWidth, fftHeight, hWidth, hHeight);
};

void thread_routine(ThreadInfo info) {
	padData(info.hChannel, info.hPadded, info.fftWidth, info.fftHeight, info.hWidth, info.hHeight, info.forward);


}

void filter_CUFFT(float* channel0, float* channel1, float* channel2, const Parameters& params)
{	
	const size_t STREAMS_NUMBER = 3;


	// transformation parameters
	const size_t
		hWidth = params.width,
		hHeight = params.height;

	size_t fftWidth = snapTransformSize(hWidth);
	size_t fftHeight = snapTransformSize(hHeight);

	// calculate transformation radius
	const size_t sqradius = square_radius(fftWidth, fftHeight, params.cutoff_frequencies);


	// vars declaration, data-related
	fComplex *dChannelPadded[STREAMS_NUMBER], *hPadded0, *hPadded1, *hPadded2;
	// vars declaration, fft-related
	//		spectrum
	fComplex *dDataSpectrum[STREAMS_NUMBER];
	//		create fft plan
	cufftHandle
		fftPlanFwd[STREAMS_NUMBER],
		fftPlanInv[STREAMS_NUMBER];

	
	
	/* ########## CALLBACK PROCEDURES, UNUSED ON WINDOWS ########## */
	//cufftCallbackStoreC h_cufftCallback;
	//// info about image 
	//ImageInfo imageInfo, *dImageInfo;
	//imageInfo.width = fftWidth;
	//imageInfo.height = fftHeight;
	//imageInfo.centerX = fftWidth >> 1;
	//imageInfo.centerY = fftHeight >> 1;
	//checkCudaErrors(cudaMalloc((void **)&dImageInfo, sizeof(ImageInfo)));
	//checkCudaErrors(cudaMemcpy(dImageInfo, &imageInfo, sizeof(ImageInfo), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyFromSymbol(
	//	&h_cufftCallback,
	//	d_cufftshiftCallback,
	//	sizeof(h_cufftCallback)
	//));

	// set callbacks
	// NOTE: ON LINUX 64bit only!!!
	/*checkCudaErrors(cufftXtSetCallback(fftPlanFwd,
		(void **)&h_cufftCallback,
		CUFFT_CB_ST_COMPLEX,
		(void**)&dImageInfo
	));
	checkCudaErrors(cufftXtSetCallback(fftPlanInv,
		(void **)&h_cufftCallback,
		CUFFT_CB_ST_COMPLEX,
		(void**)&dImageInfo
	));*/

	/* ############################################################ */

	/* ########### STREAMING ########### */
	

	// create streams
	cudaStream_t streams[STREAMS_NUMBER];
	for (int i = 0; i < STREAMS_NUMBER; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));

	// memory allocalion
	hPadded0 = (fComplex *)malloc(fftWidth * fftHeight * sizeof(fComplex));
	hPadded1 = (fComplex *)malloc(fftWidth * fftHeight * sizeof(fComplex));
	hPadded2 = (fComplex *)malloc(fftWidth * fftHeight * sizeof(fComplex));
	memset(hPadded0, 0, fftWidth * fftHeight * sizeof(fComplex));
	memset(hPadded1, 0, fftWidth * fftHeight * sizeof(fComplex));
	memset(hPadded2, 0, fftWidth * fftHeight * sizeof(fComplex));
	// gpu memory allocation
	for (int i = 0; i < STREAMS_NUMBER; i++)
	{
		checkCudaErrors(cudaMalloc((void **)&dChannelPadded[i], fftWidth * fftHeight * sizeof(fComplex)));
		checkCudaErrors(cudaMalloc((void **)&dDataSpectrum[i], fftHeight * fftWidth * sizeof(fComplex)));
	}
	
	
	printf("s %dx%d | fft %dx%d\n", hWidth, hHeight, fftWidth, fftHeight);

	// memory upload
	ThreadInfo i;
	i.fftHeight = fftHeight;
	i.fftWidth = fftWidth;
	i.hWidth = hWidth;
	i.hHeight = hHeight;
	i.forward = true;

	i.hChannel = channel0;
	i.hPadded = hPadded0;
	std::thread t0(thread_routine, i);
	i.hChannel = channel1;
	i.hPadded = hPadded1;
	std::thread t1(thread_routine, i);
	i.hChannel = channel2;
	i.hPadded = hPadded2;
	std::thread t2(thread_routine, i);
	t0.join(); t1.join(); t2.join();
/*
	padData(channel0, hPadded0, fftWidth, fftHeight, hWidth, hHeight);
	padData(channel1, hPadded1, fftWidth, fftHeight, hWidth, hHeight);
	padData(channel2, hPadded2, fftWidth, fftHeight, hWidth, hHeight);
*/
	checkCudaErrors(cudaHostRegister(hPadded0, fftWidth * fftHeight * sizeof(fComplex), cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(hPadded1, fftWidth * fftHeight * sizeof(fComplex), cudaHostRegisterPortable));
	checkCudaErrors(cudaHostRegister(hPadded2, fftWidth * fftHeight * sizeof(fComplex), cudaHostRegisterPortable));

#define DEBUG
	
	// *************************
	// *** transformations *****
	// *************************

	// create fft plans
	for (int i = 0; i < STREAMS_NUMBER; i++)
	{
		checkCudaErrors(cufftPlan2d(&fftPlanFwd[i], fftWidth, fftHeight, CUFFT_C2C));
		checkCudaErrors(cufftSetStream(fftPlanFwd[i], streams[i]));
		checkCudaErrors(cufftPlan2d(&fftPlanInv[i], fftWidth, fftHeight, CUFFT_C2C));
		checkCudaErrors(cufftSetStream(fftPlanInv[i], streams[i]));
	}
	// fft convolution
	checkCudaErrors(cudaDeviceSynchronize());
	
	
	
	

	checkCudaErrors(cudaMemcpyAsync(dChannelPadded[0], hPadded0, fftWidth * fftHeight * sizeof(cufftComplex), cudaMemcpyHostToDevice, streams[0]));
	checkCudaErrors(cudaMemcpyAsync(dChannelPadded[1], hPadded1, fftWidth * fftHeight * sizeof(cufftComplex), cudaMemcpyHostToDevice, streams[1]));
	checkCudaErrors(cudaMemcpyAsync(dChannelPadded[2], hPadded2, fftWidth * fftHeight * sizeof(cufftComplex), cudaMemcpyHostToDevice, streams[2]));

#ifdef DEBUG
	cudaEvent_t start, stop; // pomiar czasu wykonania jądra
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, streams[0]));
#endif
#pragma unroll
	for (int i = 0; i < STREAMS_NUMBER; i++)
	{
		checkCudaErrors(cufftExecC2C(fftPlanFwd[i], (cufftComplex *)dChannelPadded[i], (cufftComplex *)dDataSpectrum[i], CUFFT_FORWARD));
	}


		dim3 block(32, 32);
		dim3 grid(iDivUp(fftWidth, block.x), iDivUp(fftHeight, block.y));

		/* ###### KERNELS ###### */
		// cut frequencies kernel
		//for (int i = 0; i < STREAMS_NUMBER; i++)
		//	checkCudaErrors(cudaStreamSynchronize(streams[i]));
#pragma unroll
		for (int i = 0; i < STREAMS_NUMBER; i++)
		{
			fftShift << <grid, block, 0, streams[i] >> > (dDataSpectrum[i], fftWidth, fftHeight);
			//checkCudaErrors(cudaDeviceSynchronize());
			if (params.isLowPass)
				lowPassFilter << <grid, block, 0, streams[i] >> > (
					dDataSpectrum[i],
					fftHeight,
					fftWidth,
					hHeight,
					hWidth,
					sqradius
					);
			else
				highPassFilter << <grid, block, 0, streams[i] >> > (
					dDataSpectrum[i],
					fftHeight,
					fftWidth,
					hHeight,
					hWidth,
					sqradius
					);
			//checkCudaErrors(cudaDeviceSynchronize());
			fftShift << <grid, block, 0, streams[i] >> > (dDataSpectrum[i], fftWidth, fftHeight);
			/* ######### END KERNELS ####### */
		}

#pragma unroll
		for (int i = 0; i < STREAMS_NUMBER; i++)
		{
		//// inverse transformation
			checkCudaErrors(cufftExecC2C(fftPlanInv[i], (cufftComplex *)dDataSpectrum[i], (cufftComplex *)dChannelPadded[i], CUFFT_INVERSE));

		}

#ifdef DEBUG
	checkCudaErrors(cudaEventRecord(stop, streams[0]));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	printf("GPU (kernel) time = %.3f ms\n", elapsedTime);
#endif
	
	checkCudaErrors(cudaMemcpyAsync(hPadded0, dChannelPadded[0], fftWidth  * fftHeight * sizeof(fComplex), cudaMemcpyDeviceToHost, streams[0]));
	checkCudaErrors(cudaMemcpyAsync(hPadded1, dChannelPadded[1], fftWidth  * fftHeight * sizeof(fComplex), cudaMemcpyDeviceToHost, streams[1]));
	checkCudaErrors(cudaMemcpyAsync(hPadded2, dChannelPadded[2], fftWidth  * fftHeight * sizeof(fComplex), cudaMemcpyDeviceToHost, streams[2]));

	checkCudaErrors(cudaDeviceSynchronize());
	// TODO: add threads

	i.forward = false;

	i.hChannel = channel0;
	i.hPadded = hPadded0;
	std::thread t3(thread_routine, i);
	i.hChannel = channel1;
	i.hPadded = hPadded1;
	std::thread t4(thread_routine, i);
	i.hChannel = channel2;
	i.hPadded = hPadded2;
	std::thread t5(thread_routine, i);
	t3.join(); t4.join(); t5.join();
/*
	padData(channel0, hPadded0, fftWidth, fftHeight, hWidth, hHeight, false);
	padData(channel1, hPadded1, fftWidth, fftHeight, hWidth, hHeight, false);
	padData(channel2, hPadded2, fftWidth, fftHeight, hWidth, hHeight, false);
	*/
	// normalize
	
	// free resources
	for (int i = 0; i < STREAMS_NUMBER; i++)
	{
		checkCudaErrors(cudaStreamDestroy(streams[i]));
		checkCudaErrors(cufftDestroy(fftPlanInv[i]));
		checkCudaErrors(cufftDestroy(fftPlanFwd[i]));

		checkCudaErrors(cudaFree(dDataSpectrum[i]));

		checkCudaErrors(cudaFree(dChannelPadded[i]));
	}
	//checkCudaErrors(cudaFree(dImageInfo));
	cudaDeviceReset();
	free(hPadded0);
	free(hPadded1);
	free(hPadded2);
}