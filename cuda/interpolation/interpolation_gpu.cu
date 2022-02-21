#include "stdafx.h"
#include "interpolation_gpu.cuh"
#include "interpolation_kernel.cuh"
#include "../common/common.h"

#define BLOCK_SIZE 256

int CInterpolationGPU::prepare(SDS3D* vol) 
{
	int nErr = EC_OK;

	if (!vol)
		return EC_ERR;

	m_dim = vol->dim;
	unsigned long nSize = Common::calcDim(vol->dim);
	unsigned long nBytes = nSize * sizeof(short);

	CheckCudaErrors(cudaMalloc((void**)&m_dpVolume, nBytes));
	CheckCudaErrors(cudaMemcpy(m_dpVolume, vol->data, nBytes, cudaMemcpyHostToDevice));

	return nErr;
}

int CInterpolationGPU::run(ipSDS3D&  interp) 
{
	int nErr = EC_OK;

	unsigned long nSize = Common::calcDim(interp.dim);
	unsigned long nBytes = nSize * sizeof(short);
	unsigned long nInterpBytes = nSize * sizeof(SDSF3);

	CheckCudaErrors(cudaMalloc((void**)&m_dpInterp, nBytes));
	CheckCudaErrors(cudaMalloc((void**)&m_dpInterpSDSF3, nInterpBytes));
	CheckCudaErrors(cudaMemcpy((void**)m_dpInterpSDSF3, interp.interp, nInterpBytes, cudaMemcpyHostToDevice));

	dim3 block(BLOCK_SIZE);
	dim3 grid((interp.dim.col + BLOCK_SIZE - 1) / BLOCK_SIZE, interp.dim.row, interp.dim.hei);

	interpolationKernel << <grid, block >> >(m_dpVolume, m_dpInterp, m_dpInterpSDSF3, nSize, m_dim.col, m_dim.row, m_dim.hei);

	CheckCudaErrors(cudaMemcpy(interp.data, m_dpInterp, nBytes, cudaMemcpyDeviceToHost));
	CheckCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));

	return nErr;
}

int CInterpolationGPU::release() 
{
	int nErr = EC_OK;

	CheckCudaErrors(cudaFree(m_dpVolume));
	CheckCudaErrors(cudaFree(m_dpInterp));
	CheckCudaErrors(cudaFree(m_dpInterpSDSF3));

	return nErr;
}