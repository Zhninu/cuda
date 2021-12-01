#pragma once
#include "typedef.h"
#include "../common/timer.h"

class CBinarizeEngine
{
public:
	CBinarizeEngine();
	~CBinarizeEngine();

public:
	void setData(SDS3D* pSDS3D);
	bool binarizeOnHost(SDS3D& stBinarizeData);
	bool binarizeOnGPU(SDS3D& stBinarizeData);

private:
	const char*   m_pMoudle;
	void *		  m_pclsBinarize;
	SDS3D*		  m_pVolumeData;
	CTimer		  m_stTimer;
};