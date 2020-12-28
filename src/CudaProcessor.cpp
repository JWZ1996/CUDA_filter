#include "../include/CudaProcessor.h"
#include "../include/cudaPtr.h"
#include "helper_cuda.h"

#include <algorithm>

CudaProcessor::CudaProcessor(cv::Mat& input, IGPUFilter& filter) : gpuFilter{ filter }, inputMat(input)
{
	splitChannels();
}

 void CudaProcessor::splitChannels()
{
	 cv::split(inputMat, inputChannels);

	 std::for_each(std::begin(inputChannels), std::end(inputChannels), [this](auto& channel)
	 {
			 channelToGpu(channel);
	 });
	 
}
 void CudaProcessor::channelToGpu(cv::Mat& channel) 
 {
	 std::vector<uchar> data;
	 data.reserve(getImgSize());

	 for (int i = 0; i < inputMat.rows; i++)
	 {
		 uchar* segment_start = inputMat.data + i * inputMat.step;
		 data.insert(data.end(), segment_start, segment_start + inputMat.cols);
	 }

	 gpuChannelsData.emplace_back(&data[0]);
 }
 size_t CudaProcessor::getImgSize()
 {
	 return (inputMat.rows * inputMat.cols);
 }

 size_t CudaProcessor::getImgBSize() {
	 return getImgSize() * sizeof(float);
 }
 void CudaProcessor::apply()
 {
	 std::for_each(std::begin(gpuChannelsData), std::end(gpuChannelsData), [this](auto& channel)
     {

	     // Kanał w osobnym strumieniu
	     // Wersja batchowana

         gpuFilter.filter(channel);
		 
		 // no resource deallocation?
     });

	 merge(inputChannels, inputMat); 
 }

 void CudaProcessor::mergeChannels()
 {
	 for (size_t i = 0; i < inputChannels.size(); i++)
	 {
		 inputChannels[i] = channelToHost(gpuChannelsData[i]);
	 }

	 //mergeChannels(); 
 }
 cv::Mat CudaProcessor::channelToHost(uchar* gpuChannel)
 {

	 return cv::Mat(inputMat.rows, inputMat.cols, CV_8UC1, gpuChannel);
 }
