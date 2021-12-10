#include "binarize_kernel.cuh"

__global__ void binarizeKernel(short * volumedata, short * binarydata, int size, int thresh, int maxval)
{
	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	unsigned long threadId = blockId * blockDim.x + threadIdx.x;

	if (volumedata[threadId] < thresh)
	{
		binarydata[threadId] = volumedata[threadId];
	}
	else
	{
		binarydata[threadId] = maxval;
	}
}