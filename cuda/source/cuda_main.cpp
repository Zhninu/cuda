#include "stdafx.h"
#include "typedef.h"
#include "../binarization/binarize_engine.h"
#include "../demo/cuda_verify.cuh"
#include "../common/common.h"

int main(int argc, char **argv)
{
	if (argc < 2)
		return false;

	int nErr = EC_OK;

	Module nMType = Common::findModeType(argv[1]);

	//Binary
	if (nMType == Module_Binary
		|| nMType == Module_All)
	{
		CBinarizeEngine* pEngine = new CBinarizeEngine(NULL, 1, 255);
		pEngine->binarize(argc, argv);
	}

	//Cuda verify
	if (nMType == Module_CudaVerify
		|| nMType == Module_All)
	{
		CCudaVerify* pCudaVerify = new CCudaVerify;
		pCudaVerify->memcpyAsync(argc, argv);
	}

	return nErr;
}