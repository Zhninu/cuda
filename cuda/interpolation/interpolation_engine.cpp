#pragma once
#include "stdafx.h"
#include "interpolation_engine.h"
#include "interpolation_gpu.cuh"
#include "../common/common.h"

#define  INTERP_VOLUME_COLUME		512
#define  INTERP_VOLUME_ROW			512
#define  INTERP_VOLUME_HEIGHT		512

#define  CTV_OFFSET 1024

CInterpEngine::CInterpEngine(SDS3D* vol) : CBaseEngine(vol)
{
	m_pMoudle = LOG_BINARIZE_ENGINE_MODULE;

	if(!m_pCudaEngine)
		m_pCudaEngine = new CInterpolationGPU;
}

CInterpEngine::CInterpEngine(vdim3 dim) : CBaseEngine(dim)
{
	m_pMoudle = LOG_BINARIZE_ENGINE_MODULE;

	if (!m_pCudaEngine)
		m_pCudaEngine = new CInterpolationGPU;
}

CInterpEngine:: ~CInterpEngine()
{
	if(m_pCudaEngine)
		delete(reinterpret_cast<CInterpolationGPU *>(m_pCudaEngine));
}

int CInterpEngine::interp(int argc, char **argv)
{
	ipSDS3D h_Inter, d_Inter;
	float ratio = 0.8f;
	constructInterp(h_Inter, ratio);
	constructInterp(d_Inter, ratio);

	//interpolation
	interpHost(h_Inter);
	interpDev(d_Inter);

	//check result
	unsigned long nSize = Common::calcDim(h_Inter.dim);
	Common::campareResult(h_Inter.data, d_Inter.data, nSize);

	freeInterp(h_Inter);
	freeInterp(d_Inter);

	return true;
}

bool CInterpEngine::interpHost(ipSDS3D& ipdata)
{
	//convert array to 2D
	vdim3  dimVol = m_pVolumeData->dim;
	SDS1D  array1D;
	array1D.data = m_pVolumeData->data;
	array1D.size = Common::calcDim(dimVol);
	SDS2D  array2D;
	Common::mallocArray2D(&array2D.data, dimVol);
	array2D.dim.col = dimVol.col * dimVol.row;
	array2D.dim.row = dimVol.hei;
	Common::convertArray(&array1D, &array2D, cvArray_1DTo2D);
	//convert array to 2D

	m_stTimer.startTimer("Interpolation on host");
	unsigned long nInterpSize = Common::calcDim(ipdata.dim);
	short** pVolumeData = array2D.data;
	int nSliceWid = dimVol.col;
	int nSliceHei = dimVol.row;
	int nSliceNum = dimVol.hei;

	for (unsigned long count = 0; count < nInterpSize; count++)
	{
		SDSF3 ptInterp = ipdata.interp[count];
		short nVal = -CTV_OFFSET;
		int nx = (int)ptInterp.q1;
		int ny = (int)ptInterp.q2;
		int nz = (int)ptInterp.q3;

		if (nx < 0 || nx > nSliceWid - 1 || ny < 0 || ny > nSliceHei - 1 || nz < 0 || nz > nSliceNum - 1) 
		{
			ipdata.data[count] = nVal;
			continue;
		}

		int nz1 = nz == nSliceNum - 1 ? nz : nz + 1;
		int ny1 = ny == nSliceHei - 1 ? ny : ny + 1;
		int nx1 = nx == nSliceWid - 1 ? nx : nx + 1;

		int nyy = ny*nSliceWid;
		int nyy1 = ny1*nSliceWid;
		float fv1 = pVolumeData[nz][nyy + nx];
		float fv2 = pVolumeData[nz][nyy + nx1];
		float fv3 = pVolumeData[nz][nyy1 + nx1];
		float fv4 = pVolumeData[nz][nyy1 + nx];
		float fv5 = pVolumeData[nz1][nyy + nx];
		float fv6 = pVolumeData[nz1][nyy + nx1];
		float fv7 = pVolumeData[nz1][nyy1 + nx1];
		float fv8 = pVolumeData[nz1][nyy1 + nx];
		float fsx = ptInterp.q1 - (float)nx;
		float fsy = ptInterp.q2 - (float)ny;
		float fsz = ptInterp.q3 - (float)nz;
		float fz1 = fv1 + (fv5 - fv1)*fsz;
		float fz4 = fv4 + (fv8 - fv4)*fsz;
		float fx1 = fz1 + (fv2 + (fv6 - fv2)*fsz - fz1)*fsx;
		nVal = fx1 + (fz4 + (fv3 + (fv7 - fv3)*fsz - fz4)*fsx - fx1)*fsy + 0.5f;
		nVal = nVal > -CTV_OFFSET ? nVal : -CTV_OFFSET;
		nVal = nVal < 3072 ? nVal : 3071;

		ipdata.data[count] = nVal;
	}
	m_stTimer.stopTimer("Interpolation on host");

	Common::freeArray2D(&array2D.data, dimVol);

	return true;
}

bool CInterpEngine::interpDev(ipSDS3D& ipdata)
{
	bool bRet = false;

	do
	{
		if (!m_pVolumeData || !m_pCudaEngine)
		{
			log_error(m_pMoudle, LogFormatA_A("Cuda engine or volume data is Null!").c_str());
			break;
		}

		m_stTimer.startTimer("Interpolation on GPU");
		reinterpret_cast<CInterpolationGPU *>(m_pCudaEngine)->prepare(m_pVolumeData);
		reinterpret_cast<CInterpolationGPU *>(m_pCudaEngine)->run(ipdata);
		reinterpret_cast<CInterpolationGPU *>(m_pCudaEngine)->release();
		m_stTimer.stopTimer("Interpolation on GPU");

		bRet = true;

	} while (0);

	return true;
}

void CInterpEngine::constructInterp(ipSDS3D& ipdata, float ratio)
{
	ipdata.dim = m_pVolumeData->dim;
	unsigned long nSize = Common::calcDim(ipdata.dim);
	ipdata.data = (short*)malloc(nSize * sizeof(short));
	ipdata.interp = (SDSF3*)malloc(nSize * sizeof(SDSF3));
	memset(ipdata.data, 0, nSize * sizeof(short));
	memset(ipdata.interp, 0, nSize * sizeof(SDSF3));

	for (unsigned long hei = 0; hei < ipdata.dim.hei; hei++)
	{
		for (unsigned long row = 0; row < ipdata.dim.row; row++)
		{
			for (unsigned long col = 0; col < ipdata.dim.col; col++)
			{
				unsigned long unIdx = col + row * ipdata.dim.col + hei * ipdata.dim.col * ipdata.dim.row;
				ipdata.interp[unIdx].q1 = col + ratio;
				ipdata.interp[unIdx].q2 = row + ratio;
				ipdata.interp[unIdx].q3 = hei + ratio;
			}
		}
	}
}

void CInterpEngine::freeInterp(ipSDS3D& ipdata)
{
	if(ipdata.data)
		free(ipdata.data);

	if (ipdata.interp)
		free(ipdata.interp);
}