#pragma once
#include "typedef.h"
#include "../common/timer.h"

class CBaseEngine
{
public:
	CBaseEngine(SDS3D* vol);
	CBaseEngine(vdim3  dim);
	~CBaseEngine();

private:
	void constructVol(vdim3 dim);
	void destroyVol();

protected:
	const char*   m_pMoudle;
	void *		  m_pCudaEngine;
	SDS3D*		  m_pVolumeData;
	CTimer		  m_stTimer;

private:
	bool		  m_bCreateVol;
};