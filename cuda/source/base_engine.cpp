#include "stdafx.h"
#include "base_engine.h"
#include "../common/common.h"
#include "../common/log_message.h"

#define  CONSTRUCT_VOLUME_COLUME		512
#define  CONSTRUCT_VOLUME_ROW			512
#define  CONSTRUCT_VOLUME_HEIGHT		64

CBaseEngine::CBaseEngine(SDS3D* vol)
	: m_pMoudle("")
	, m_pCudaEngine(NULL)
	, m_pVolumeData(vol)
	, m_bCreateVol(false)
{
	if (!m_pVolumeData)
	{
		vdim3 dim(CONSTRUCT_VOLUME_COLUME, CONSTRUCT_VOLUME_ROW, CONSTRUCT_VOLUME_HEIGHT);
		constructVol(dim);
	}
	else
		m_pVolumeData = vol;
}

CBaseEngine::CBaseEngine(vdim3 dim)
	: m_pMoudle("")
	, m_pCudaEngine(NULL)
	, m_pVolumeData(NULL)
	, m_bCreateVol(false)
{
	if (!m_pVolumeData)
		constructVol(dim);
}

CBaseEngine::~CBaseEngine()
{
	if (m_bCreateVol)
		destroyVol();
}

void CBaseEngine::constructVol(vdim3 dim)
{
	m_pVolumeData = new SDS3D;
	m_pVolumeData->data = NULL;
	m_pVolumeData->dim = dim;
	m_bCreateVol = Common::mallocArray1D(&m_pVolumeData->data, dim);
	if (m_bCreateVol)
	{
		Common::constructArray(m_pVolumeData->data, Common::calcDim(m_pVolumeData->dim));
	}
	else
	{
		delete m_pVolumeData;
		m_pVolumeData = NULL;
	}
}

void CBaseEngine::destroyVol()
{
	m_bCreateVol = Common::freeArray1D(&m_pVolumeData->data);
	delete m_pVolumeData;
	m_pVolumeData = NULL;
}