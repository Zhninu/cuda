#include "typedef.h"
#include "../common/timer.h"

class CBinarizeGPU
{
public:
	int prepare(SDS3D* volumedata);
	int run(binSDS3D&  binarydata);
	int release();

private:
	CTimer	m_stTimer;
	short*	m_dpVolume;
	short*	m_dpBinarize;
};