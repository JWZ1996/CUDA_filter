#include "FilterMask.h"

inline unsigned char FilterMatrix::getPixelValue(std::valarray<int>&& input) const
{
	// Każdy el. input zostaje przemnożony przez odpowiadający mu el. z matrix
	int matrixSum{ matrix.sum() };

	input *= matrix;

	if (matrixSum != 0)
	{
		input /= matrixSum;
	}

	// Zapobieganie overflow sumy tablicy input
	return std::clamp(input.sum(), 0, static_cast<int>(pow(2, bit_depth) - 1.0f));
}
