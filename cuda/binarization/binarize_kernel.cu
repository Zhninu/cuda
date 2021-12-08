#include "binarize_kernel.cuh"

__global__ void binarizeKernel(short * volumedata, short * binarydata, int size, int thresh, int maxval)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;

	if (volumedata[threadId] < thresh)
	{
		binarydata[threadId] = volumedata[threadId];
	}
	else
	{
		binarydata[threadId] = maxval;
	}
}