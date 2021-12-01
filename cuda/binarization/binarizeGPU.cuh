#include "typedef.h"
#include "../common/timer.h"

class CBinarizeGPU
{
public:
	int prepare(SDS3D *pVolumeData);
	int run(SDS3D& stBinarizeData);
	int release();

private:
	CTimer	m_stTimer;
	short*	m_dpVolume;
	short*	m_dpBinarize;
};