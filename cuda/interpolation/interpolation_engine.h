#pragma once
#include "typedef.h"
#include "../common/timer.h"

class CInterpEngine
{
public:
	CInterpEngine(SDS3D* vol);
	CInterpEngine(vdim3  dim);
	~CInterpEngine();

public:
	int  interp(int argc, char **argv);

private:
	bool interpHost(ipSDS3D& ipdata);
	bool interpDev(ipSDS3D& ipdata);
	void constructInterp(ipSDS3D& ipdata, float ratio);
	void freeInterp(ipSDS3D& ipdata);

private:
	const char*   m_pMoudle;
	void *		  m_pclsInterp;
	SDS3D*		  m_pVolumeData;
	bool		  m_bCreateVol;
	CTimer		  m_stTimer;
};