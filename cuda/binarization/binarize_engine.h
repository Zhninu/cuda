#pragma once
#include "typedef.h"
#include "base_engine.h"
#include "../common/timer.h"

class CBinarizeEngine : public CBaseEngine
{
public:
	CBinarizeEngine(SDS3D* vol, int thresh, int maxval);
	CBinarizeEngine(vdim3  dim, int thresh, int maxval);
	~CBinarizeEngine();

public:
	int  binarize(int argc, char **argv);

private: 
	bool binarizeHost(binSDS3D& binarydata);
	bool binarizeDev(binSDS3D& binarydata);

private:
	int			  m_nThresh;
	int			  m_nMaxVal;
};