#include "stdafx.h"
#include "binarize_gpu.cuh"
#include "binarize_kernel.cuh"
#include "../common/common.h" 

#define BLOCK_SIZE 256

int CBinarizeGPU::prepare(SDS3D* volumedata)
{
	int nErr = EC_OK; 

	if (!volumedata)
		return EC_ERR;

	unsigned long nSize = Common::calcDim(volumedata->dim);
	unsigned long nBytes = nSize * sizeof(short);

	CheckCudaErrors(cudaMalloc((void**)&m_dpVolume, nBytes));
	CheckCudaErrors(cudaMalloc((void**)&m_dpBinarize, nBytes));
	CheckCudaErrors(cudaMemcpy(m_dpVolume, volumedata->data, nBytes, cudaMemcpyHostToDevice));

	return nErr;
}

int CBinarizeGPU::run(binSDS3D& binarydata)
{
	int nErr = EC_OK;

	unsigned long nSize = Common::calcDim(binarydata.dim);
	unsigned long nBytes = nSize * sizeof(short);

	dim3 block(BLOCK_SIZE);
	dim3 grid((binarydata.dim.col + BLOCK_SIZE - 1) / BLOCK_SIZE, binarydata.dim.row, binarydata.dim.hei);
	binarizeKernel << <grid, block >> >(m_dpVolume, m_dpBinarize, nSize, binarydata.thresh, binarydata.maxval);
	CheckCudaErrors(cudaMemcpy(binarydata.data, m_dpBinarize, nBytes, cudaMemcpyDeviceToHost));
	CheckCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));

	return nErr;
}

int CBinarizeGPU::release() 
{
	int nErr = EC_OK;

	CheckCudaErrors(cudaFree(m_dpVolume));
	CheckCudaErrors(cudaFree(m_dpBinarize));

	return nErr;
}