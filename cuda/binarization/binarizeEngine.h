#pragma once
#include "typedef.h"
#include "../common/timer.h"

#define  VOLUME_COLUME		512
#define  VOLUME_ROW			512
#define  VOLUME_HEIGHT		512

class CBinarizeEngine
{
public:
	CBinarizeEngine();
	~CBinarizeEngine();

public:
	void setData(SDS3D* pSDS3D);
	void setDataDim(VolDim dim);
	int  Binarize();

private: 
	bool createVolume();
	void freeVolume();
	bool binarizeHost(SDS3D& stBinarizedVol);
	bool binarizeDev(SDS3D& stBinarizedVol);

private:
	const char*   m_pMoudle;
	void *		  m_pclsBinarize;
	SDS3D*		  m_pVolumeData;
	VolDim		  m_stVolDim;
	CTimer		  m_stTimer;
};