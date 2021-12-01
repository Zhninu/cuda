#include "binarizeKernel.cuh"

__global__ void binarizeKernel(short *pVolData, short *pBinData, int nSize)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;

	if (pVolData[threadId] < 20)
	{
		pBinData[threadId] = 0;
	}
	else
	{
		pBinData[threadId] = 1;
	}
}