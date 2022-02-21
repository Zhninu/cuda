#include "typedef.h"
#include "../common/timer.h"

class CInterpolationGPU
{
public:
	int prepare(SDS3D* vol);
	int run(ipSDS3D&  interp);
	int release();

private:
	CTimer	m_stTimer;
	short*	m_dpVolume;
	short*	m_dpInterp;
	SDSF3*  m_dpInterpSDSF3;
	vdim3   m_dim;
};