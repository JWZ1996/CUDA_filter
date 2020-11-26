#include "CudaProcessor.h"
#include "cudaPtr.h"
#include "helper_cuda.h"

#include <algorithm>

CudaExecutor::CudaExecutor(cv::Mat& input, IGPUFilter& filter) : gpuFilter{ filter }, inputMat { input }
{
	splitChannels();
}

 void CudaExecutor::splitChannels()
{
	 split(inputMat, inputChannels);

	 std::for_each(std::begin(inputChannels), std::end(inputChannels), [this](auto& channel)
	 {
			 channelToGpu(channel);
	 });
	 
}
 void CudaExecutor::channelToGpu(cv::Mat& channel)
 {
	 std::vector<uchar> data;
	 data.reserve(getImgSize());

	 for (int i = 0; i < inputMat.rows; i++)
	 {
		 uchar* segment_start = inputMat.data + i * inputMat.step;
		 data.insert(data.end(), segment_start, segment_start + inputMat.cols);
	 }

	 uchar* memStart{ nullptr };
	 checkCudaErrors(cudaMalloc(&memStart, getImgSize()));
	 checkCudaErrors(cudaMemcpy(memStart, data.data(), getImgSize(), cudaMemcpyHostToDevice));

	 gpuChannelsData.emplace_back(memStart);
 }
 size_t CudaExecutor::getImgSize()
 {
	 return (inputMat.rows * inputMat.cols);
 }
 void CudaExecutor::apply()
 {
	 std::for_each(std::begin(gpuChannelsData), std::end(gpuChannelsData), [this](auto& channel)
     {

	     // Kanał w osobnym strumieniu
	     // Wersja batchowana
         gpuFilter.filter(channel);
     });

	 merge(inputChannels, inputMat);
 }

 void CudaExecutor::mergeChannels()
 {
	 for (size_t i = 0; i < inputChannels.size(); i++)
	 {
		 inputChannels[i] = channelToHost(gpuChannelsData[i]);
	 }

	 mergeChannels();
 }
 cv::Mat CudaExecutor::channelToHost(uchar* gpuChannel)
 {
	 std::vector<uchar> data;
	 data.reserve(getImgSize());

	 checkCudaErrors(cudaMemcpy(data.data(), gpuChannel, getImgSize(), cudaMemcpyDeviceToHost));

	 return cv::Mat(inputMat.rows, inputMat.cols, CV_8UC1, data.data());
 }
