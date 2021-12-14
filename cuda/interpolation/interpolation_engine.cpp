#pragma once
#include "stdafx.h"
#include "interpolation_engine.h"
#include "interpolation_gpu.cuh"
#include "../common/common.h"

#define  INTERP_VOLUME_COLUME		512
#define  INTERP_VOLUME_ROW			512
#define  INTERP_VOLUME_HEIGHT		1024

#define  CTV_OFFSET 1024

CInterpEngine::CInterpEngine(SDS3D* volumedata) 
	: m_pMoudle(LOG_BINARIZE_ENGINE_MODULE)
	, m_pclsInterp(NULL)
	, m_pVolumeData(volumedata)
	, m_bCreateVol(false)
{
	if(!m_pclsInterp)
		m_pclsInterp = new CInterpolationGPU;

	if (!volumedata) 
	{
		vdim3 voldim(INTERP_VOLUME_COLUME, INTERP_VOLUME_ROW, INTERP_VOLUME_HEIGHT);
		m_bCreateVol = Common::mallocVolume(&m_pVolumeData, voldim);
	}
}

CInterpEngine::CInterpEngine(vdim3  dim) 
	: m_pMoudle(LOG_BINARIZE_ENGINE_MODULE)
	, m_pclsInterp(NULL)
	, m_pVolumeData(NULL)
	, m_bCreateVol(false)
{
	if (!m_pclsInterp)
		m_pclsInterp = new CInterpolationGPU;

	m_bCreateVol = Common::mallocVolume(&m_pVolumeData, dim);
}

CInterpEngine:: ~CInterpEngine()
{
	if(m_pclsInterp)
		delete(reinterpret_cast<CInterpolationGPU *>(m_pclsInterp));

	if (m_bCreateVol)
		m_bCreateVol = Common::freeVolume(&m_pVolumeData);
}

int CInterpEngine::interp() 
{

}

bool CInterpEngine::interpHost(interSDS3D& interpdata)
{
	//convert array to 2D
	vdim3  dimVol = m_pVolumeData->dim;
	SDS1D  array1D;
	array1D.data = m_pVolumeData->data;
	array1D.size = Common::calcDim(dimVol);
	SDS2D  array2D;
	Common::allocArray2D(array2D.data, dimVol);
	array2D.dim.col = dimVol.col * dimVol.row;
	array2D.dim.row = dimVol.hei;
	Common::convertArray2D(&array1D, &array2D, cvArray_1DTo2D);

	unsigned long nInterpSize = Common::calcDim(interpdata.dim);
	short** pVolumeData = array2D.data;
	int nSliceWid = dimVol.col;
	int nSliceHei = dimVol.row;
	int nSliceNum = dimVol.hei;

	for (unsigned long count = 0; count < nInterpSize; count++)
	{
		SDSF3 ptInterp = interpdata.interp[count];
		short nVal = -CTV_OFFSET;
		int nx = (int)ptInterp.q1;
		int ny = (int)ptInterp.q2;
		int nz = (int)ptInterp.q3;

		if (nx < 0 || nx > nSliceWid - 1 || ny < 0 || ny > nSliceHei - 1 || nz < 0 || nz > nSliceNum - 1) 
			return nVal;

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

		interpdata.data[count] = nVal;
	}
}

bool CInterpEngine::interpDev(interSDS3D& interpdata)
{


}