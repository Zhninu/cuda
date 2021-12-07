#pragma once
#include "stdafx.h"
#include "binarizeEngine.h"
#include "binarizeGPU.cuh"
#include "../common/common.h"
#include "../common/logMessage.h"

#define LOG_BINARIZEENGINE_MODULE	"BinarizeEngine"

CBinarizeEngine::CBinarizeEngine()
{
	m_pMoudle = LOG_BINARIZEENGINE_MODULE;
	m_pVolumeData = NULL;
	m_stVolDim.nCol = VOLUME_COLUME;
	m_stVolDim.nRow = VOLUME_ROW;
	m_stVolDim.nHei = VOLUME_HEIGHT;
	m_pclsBinarize = new CBinarizeGPU;
}

CBinarizeEngine::~CBinarizeEngine()
{
	delete(reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize));
}

void CBinarizeEngine::setData(SDS3D* pSDS3D) 
{
	if (!pSDS3D)
		return;

	m_pVolumeData = pSDS3D;
}

void CBinarizeEngine::setDataDim(VolDim dim)
{
	m_stVolDim.nCol = dim.nCol;
	m_stVolDim.nRow = dim.nRow;
	m_stVolDim.nHei = dim.nHei;
}

int CBinarizeEngine::Binarize() 
{
	int	  nErr = EC_OK;
	bool  bCreate = false;

	if (m_pVolumeData == NULL) 
	{
		bCreate = createVolume();
	}

	SDS3D BinarizedVol;
	SDS3D BinarizedVolGPU;
	BinarizedVol.nDim = m_pVolumeData->nDim;
	BinarizedVolGPU.nDim = m_pVolumeData->nDim;

	int nSize = calcDimSize(m_pVolumeData->nDim);
	int nBytes = nSize * sizeof(short);

	BinarizedVol.pData = (short*)malloc(nBytes);
	BinarizedVolGPU.pData = (short*)malloc(nBytes);

	binarizeHost(BinarizedVol);
	binarizeDev(BinarizedVolGPU);

	campareResult(BinarizedVol.pData, BinarizedVolGPU.pData, nSize);

	if(bCreate)
		freeVolume();

	free(BinarizedVol.pData);
	free(BinarizedVolGPU.pData);

	return nErr;
}

bool CBinarizeEngine::createVolume() 
{
	bool bCreate = false;

	if (m_pVolumeData)
		return bCreate;

	m_pVolumeData = new SDS3D;
	m_pVolumeData->nDim.nCol = m_stVolDim.nCol;
	m_pVolumeData->nDim.nRow = m_stVolDim.nRow;
	m_pVolumeData->nDim.nHei = m_stVolDim.nHei;

	int nSize = calcDimSize(m_pVolumeData->nDim);
	int nBytes = nSize * sizeof(short);

	m_pVolumeData->pData = (short*)malloc(nBytes);
	initRandData(m_pVolumeData->pData, nSize);

	bCreate = true;

	return true;
}

void CBinarizeEngine::freeVolume()
{
	free(m_pVolumeData->pData);
}

bool CBinarizeEngine::binarizeHost(SDS3D& stBinarizedVol)
{
	bool bRet = false;

	do
	{
		if (!m_pVolumeData)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data is Null!").c_str());
			break;
		}

		int nVolSize = calcDimSize(m_pVolumeData->nDim);
		int nBinSize = calcDimSize(stBinarizedVol.nDim);

		if (nVolSize != nBinSize)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data size is different binarize data, %d / %d!", nVolSize, nBinSize).c_str());
			break;
		}

		m_stTimer.startTimer("Binarize on host");
		for (int z = 0; z < m_pVolumeData->nDim.nHei; z++)
		{
			for (int y = 0; y < m_pVolumeData->nDim.nRow; y++)
			{
				for (int x = 0; x < m_pVolumeData->nDim.nCol; x++)
				{
					int nIdx = x + m_pVolumeData->nDim.nCol * y + m_pVolumeData->nDim.nCol * m_pVolumeData->nDim.nRow * z;

					if (m_pVolumeData->pData[nIdx] < 20)
					{
						stBinarizedVol.pData[nIdx] = 0;
					}
					else
					{
						stBinarizedVol.pData[nIdx] = 1;
					}
				}
			}
		}
		m_stTimer.stopTimer("Binarize on host");

		bRet = true;

	} while (0);

	return bRet;
}

bool CBinarizeEngine::binarizeDev(SDS3D& stBinarizedVol)
{
	bool bRet = false;

	do
	{
		if (!m_pVolumeData)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data is Null!").c_str());
			break;
		}

		int nVolSize = calcDimSize(m_pVolumeData->nDim);
		int nBinSize = calcDimSize(stBinarizedVol.nDim);

		if (nVolSize != nBinSize)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data size is different binarize data, %d / %d!", nVolSize, nBinSize).c_str());
			break;
		}

		m_stTimer.startTimer("Binarize on GPU");
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->prepare(m_pVolumeData);
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->run(stBinarizedVol);
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->release();
		m_stTimer.stopTimer("Binarize on GPU");

		bRet = true;

	} while (0);

	return bRet;
}