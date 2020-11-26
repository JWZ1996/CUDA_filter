#pragma once
#include <memory>
#include <functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "cstdlib"

namespace CudaPtr
{
	template<typename T>
	using unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

	template<typename T>
	std::function<void(T*)> deleter = [](T* removed_ptr) { free(removed_ptr); };

	template<typename T>
	unique_ptr<T>&& make_unique() 
	{
		T* ptr {nullptr};
		checkCudaErrors(cudaMalloc<T>(&ptr, sizeof(T)));

		return std::move(unique_ptr<T>(new(ptr) T(), deleter<T>));
	}
	template<typename T>
	unique_ptr<T>&& make_unique(size_t bytes_count)
	{
		T* ptr{ (T*)malloc(sizeof(T) * bytes_count) };
		

		return std::move(unique_ptr<T>(new(ptr) T[bytes_count], deleter<T>));
	}
};