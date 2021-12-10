#pragma once
#include "typedef.h"
#include "../common/timer.h"

#define  BINARY_VOLUME_COLUME		512
#define  BINARY_VOLUME_ROW			512
#define  BINARY_VOLUME_HEIGHT		1024

class CBinarizeEngine
{
public:
	CBinarizeEngine(SDS3D* volumedata, int thresh, int maxval);
	CBinarizeEngine(vdim3  dim, int thresh, int maxval);
	~CBinarizeEngine();

public:
	int  binarize();

private: 
	bool binarizeHost(binSDS3D& binarydata);
	bool binarizeDev(binSDS3D& binarydata);

private:
	const char*   m_pMoudle;
	void *		  m_pclsBinarize;
	bool		  m_bCreateVol;
	SDS3D*		  m_pVolumeData;
	int			  m_nThresh;
	int			  m_nMaxVal;
	CTimer		  m_stTimer;
};