#pragma once
#pragma once
#include "typedef.h"
#include "../common/timer.h"

#define  BINARY_VOLUME_COLUME		512
#define  BINARY_VOLUME_ROW			512
#define  BINARY_VOLUME_HEIGHT		512

class CInterpEngine
{
public:
	CInterpEngine(SDS3D* volumedata);
	~CInterpEngine();

public:
	int  interp(SDS3D* interpdata);

private:
	bool interpHost(SDS3D& interpdata);
	bool interpDev(SDS3D& interpdata);

private:
	const char*   m_pMoudle;
	void *		  m_pclsInterp;
	SDS3D*		  m_pVolumeData;
	CTimer		  m_stTimer;
};