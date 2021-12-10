#pragma once
#include "typedef.h"
#include "../common/timer.h"

class CInterpEngine
{
public:
	CInterpEngine(SDS3D* volumedata);
	CInterpEngine(vdim3  dim);
	~CInterpEngine();

public:
	int  interp();

private:
	bool interpHost(interSDS3D& interpdata);
	bool interpDev(interSDS3D& interpdata);

private:
	const char*   m_pMoudle;
	void *		  m_pclsInterp;
	SDS3D*		  m_pVolumeData;
	bool		  m_bCreateVol;
	CTimer		  m_stTimer;
};