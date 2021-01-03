// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


typedef float2 fComplex;

__constant__  float2 dZero;

__global__ void padData(
	float *dDst,
	float *dSrc,
	int fftH,
	int fftW,
	int dataH,
	int dataW
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderH = dataH;
	const int borderW = dataW;

	if (y < fftH && x < fftW)
	{
		int dy, dx;

		if (y < dataH)
		{
			dy = y;
		}

		if (x < dataW)
		{
			dx = x;
		}

		if (y >= dataH && y < borderH)
		{
			dy = dataH - 1;
		}

		if (x >= dataW && x < borderW)
		{
			dx = dataW - 1;
		}

		if (y >= borderH)
		{
			dy = 0;
		}

		if (x >= borderW)
		{
			dx = 0;
		}

		dDst[y * fftW + x] = dSrc[dy * dataW + dx];
	}
}


// function for cutting frequencies from image
__global__ void cutFrequencies(
	fComplex *dData,
	int fftH,
	int fftW,
	int kernelH,
	int kernelW
) 
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	// calculate center of the image
	const int centerX = fftW / 2;
	const int centerY = fftH / 2;

	if (abs(centerX - x) > kernelW || abs(centerY - y) > kernelH)
		dData[y * fftW + x] = dZero;


}

// round up
inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


extern "C"
void filter_CUFFT(float* hChannel, size_t hWidth, size_t hHeight, size_t kWidth, size_t kHeight); // main filtering function



void filter_CUFFT(float* hChannel, float* hResults, size_t hWidth, size_t hHeight, size_t kWidth, size_t kHeight)
{
	// vars declaration, data-related
	float*
		dChannel, // data copied to gpu
		*dPadded,  //  data padded with kernel
		*dResults // data copied to gpu
		;

	// vars declaration, fft-related

	//		create fft plan
	cufftHandle
		fftPlanFwd, 
		fftPlanInv;

	//		spectrum
	fComplex *dDataSpectrum;



	size_t fftWidth = hWidth + kWidth + (hWidth % kWidth);
	size_t fftHeight = hHeight + kHeight + (hHeight % kHeight);
	
	// memory allocation
	checkCudaErrors(cudaMalloc((void **)&dChannel, hWidth   * hHeight * sizeof(float)));
	

	checkCudaErrors(cudaMalloc((void **)&dPadded, fftHeight * fftWidth * sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&dDataSpectrum, fftHeight * (fftWidth / 2 + 1) * sizeof(fComplex)));

	// memory upload
	const float2 hZero = make_float2(0.0f, 0.0f);
	cudaMemcpyToSymbol(&dZero, &hZero, sizeof(float2));
	checkCudaErrors(cudaMemcpy(dChannel, hChannel, hWidth   * hHeight * sizeof(float), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemset(dPadded, 0, fftHeight * fftWidth * sizeof(float)));


	dim3 block(32, 32);
	dim3 grid(iDivUp(fftWidth, block.x), iDivUp(fftHeight, block.y));

	padData<<<grid, block>>>(
		dPadded,
		dChannel,
		fftHeight,
		fftWidth,
		hHeight,
		hWidth
		);

	// *************************
	// *** transformations *****
	// *************************

	// create fft plans
	checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftHeight, fftWidth, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&fftPlanInv, fftHeight, fftWidth, CUFFT_C2R));

	// fft convolution
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)dPadded, (cufftComplex *)dDataSpectrum));

	// cut frequencies kernel
	cutFrequencies << <grid, block >> > (
		dDataSpectrum,
		fftHeight,
		fftWidth,
		hHeight,
		hWidth
		);

	// inverse transformation
	checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)dDataSpectrum, (cufftReal *)dPadded));

	checkCudaErrors(cudaDeviceSynchronize());
	
	printf("...reading back GPU convolution results\n");
	checkCudaErrors(cudaMemcpy(hResults, dResults, fftHeight * fftWidth * sizeof(float), cudaMemcpyDeviceToHost));

	// free resources
	checkCudaErrors(cufftDestroy(fftPlanInv));
	checkCudaErrors(cufftDestroy(fftPlanFwd));

	checkCudaErrors(cudaFree(dDataSpectrum));

	checkCudaErrors(cudaFree(dPadded));
	checkCudaErrors(cudaFree(dChannel));
	
}