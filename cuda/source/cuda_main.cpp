#include "stdafx.h"
#include "typedef.h"
#include "../binarization/binarize_engine.h"
#include "../common/common.h"

int main() 
{
	int nErr = EC_OK;

	//binary
	CBinarizeEngine* pEngine = new CBinarizeEngine(NULL, 1, 255);
	pEngine->binarize();

	return nErr;
}