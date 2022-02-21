#pragma once
#include "typedef.h"
#include "base_engine.h"
#include "../common/timer.h"

class CInterpEngine : public CBaseEngine
{
public:
	CInterpEngine(SDS3D* vol);
	CInterpEngine(vdim3 dim);
	~CInterpEngine();

public:
	int  interp(int argc, char **argv);

private:
	bool interpHost(ipSDS3D& ipdata);
	bool interpDev(ipSDS3D& ipdata);
	void constructInterp(ipSDS3D& ipdata, float ratio);
	void freeInterp(ipSDS3D& ipdata);
};