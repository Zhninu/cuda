#include "stdafx.h"
#include "binarizeGPU.cuh"
#include "binarizeKernel.cuh"
#include "../common/common.h" 

#define BLOCK_SIZE 256

int CBinarizeGPU::prepare(SDS3D *pVolumeData) 
{
	int nErr = EC_OK; 

	if (!pVolumeData) 
		return EC_ERR;

	int nVolSize = calcDimSize(pVolumeData->nDim);
	int nBytes = nVolSize * sizeof(short);

	CheckCudaErrors(cudaMalloc((void**)&m_dpVolume, nBytes));
	CheckCudaErrors(cudaMalloc((void**)&m_dpBinarize, nBytes));
	CheckCudaErrors(cudaMemcpyAsync(m_dpVolume, pVolumeData->pData, nBytes, cudaMemcpyHostToDevice));

	return nErr;
}

int CBinarizeGPU::run(SDS3D& stBinarizedData)
{
	int nErr = EC_OK;

	int nSize = calcDimSize(stBinarizedData.nDim);
	int nBytes = nSize * sizeof(short);

	dim3 block(BLOCK_SIZE);
	dim3 grid((stBinarizedData.nDim.nCol + BLOCK_SIZE - 1) / BLOCK_SIZE, stBinarizedData.nDim.nRow, stBinarizedData.nDim.nHei);
	binarizeKernel << <grid, block >> >(m_dpVolume, m_dpBinarize, nSize);
	CheckCudaErrors(cudaMemcpyAsync(stBinarizedData.pData, m_dpBinarize, nBytes, cudaMemcpyDeviceToHost));
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