#ifndef __KERNELS_CU__
#define __KERNELS_CU__



typedef float2 fComplex;

__constant__  float2 dZero;
//__constant__  float dFact;


// function for in-place cutting frequencies from image
__global__ void cutFrequencies(
	fComplex *dData,
	int fftH,
	int fftW,
	int imageW,
	int imageH,
	const size_t sqradius
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (x < fftW && y < fftH) {
		const int centerX = fftW >> 1;
		const int centerY = fftH >> 1;
		const int coorY = (y - centerY);
		const int coorX = (x - centerX);
		if (coorX * coorX + coorY * coorY > sqradius) {
			const int ind = y * fftW + x;
			dData[ind].x = 0.0f;
			dData[ind].y = 0.0f;
		}
	}
}

__global__ void fftShift(
	fComplex *dData,
	int fftH,
	int fftW
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	const int ind = y * fftW + x;
	int shiftX, shiftY;
	const int centerX = fftW >> 1;
	const int centerY = fftH >> 1;
	if (x < fftW && y < centerY) {

		// 1q to 4q
		if (x < centerX) {
			shiftX = x + centerX;
			shiftY = y + centerY;
		}
		// 2q to 3q
		else if (x >= centerX) {
			shiftX = x - centerX;
			shiftY = y + centerY;
		}

		//const int indShift = shiftY * imageW + shiftX;
		const fComplex v = fComplex{ dData[ind].x, dData[ind].y };
		const int indShift = fftW * shiftY +  shiftX;
		dData[ind].x = dData[indShift].x;
		dData[ind].y = dData[indShift].y;

		dData[indShift].x = v.x;
		dData[indShift].y = v.y;

		}
}


#endif