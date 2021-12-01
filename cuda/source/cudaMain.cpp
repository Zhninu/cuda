#include "stdafx.h"
#include "typedef.h"
#include "../binarization/binarizeEngine.h"
#include "../common/common.h"

#define  VOLUME_WID    512
#define  VOLUME_HEI    512
#define  VOLUME_NUM    512
 
int main() 
{
	int nErr = EC_OK;

	SDS3D VolumeData;
	VolumeData.nWid = VOLUME_WID;
	VolumeData.nHei = VOLUME_HEI;
	VolumeData.nNum = VOLUME_NUM;

	SDS3D BinarizeData = VolumeData;
	SDS3D BinarizeDataGPU = VolumeData; 

	int nSize = VolumeData.nWid * VolumeData.nHei * VolumeData.nNum;
	int nBytes = nSize * sizeof(short);

	VolumeData.pVolumeData = (short*)malloc(nBytes);
	BinarizeData.pVolumeData = (short*)malloc(nBytes);
	BinarizeDataGPU.pVolumeData = (short*)malloc(nBytes);

	initRandData(VolumeData.pVolumeData, nSize);

	CBinarizeEngine* BinEngine = new CBinarizeEngine;
	BinEngine->setData(&VolumeData);
	BinEngine->binarizeOnHost(BinarizeData);
	BinEngine->binarizeOnGPU(BinarizeDataGPU);

	campareResult(BinarizeData.pVolumeData, BinarizeDataGPU.pVolumeData, nSize);

	free(VolumeData.pVolumeData);
	free(BinarizeData.pVolumeData);
	free(BinarizeDataGPU.pVolumeData);

	return nErr;
}