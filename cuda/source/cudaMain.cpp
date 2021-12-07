#include "stdafx.h"
#include "typedef.h"
#include "../binarization/binarizeEngine.h"
#include "../common/common.h"

int main() 
{
	int nErr = EC_OK;

	CBinarizeEngine* pEngine = new CBinarizeEngine;
	VolDim dimVol;
	dimVol.nCol = VOLUME_COLUME;
	dimVol.nRow = VOLUME_ROW;
	dimVol.nHei = VOLUME_HEIGHT;

	pEngine->setDataDim(dimVol);
	pEngine->Binarize();

	return nErr;
}