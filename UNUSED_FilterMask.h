#pragma once
#include <valarray>
#include <algorithm>

class FilterMatrix
{
	unsigned bit_depth {sizeof(char)}; 

	// std::valarray to kontener, w którym zaimplementowano operatory, działające per element
	// Na przykład a + b, gdzie a i b to instancje std::valarray, zwraca kontener, w którym kazdy element jest sumą elementów z kontenerów a i b na odpowiadających sobie pozycjach
	std::valarray<int> matrix
	{
		-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1,
		-1,-1, 8,-1,-1,
		-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1
	};

public:

	unsigned char getPixelValue(std::valarray<int>&& input)  const;
};
