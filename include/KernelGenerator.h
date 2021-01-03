#pragma once
class KernelGenerator
{
public:
	static void generateMedianKernel(float*, int);
	static void generateGaussianKernel(float*, int, float);
	static void returnPrewittHorizontalKernel(float*);
	static void returnPrewittVerticalKernel(float*);
	static void returnSobelHorizontalKernel(float*);
	static void returnSobelVerticalKernel(float*);
};

