#pragma once
#include "stdafx.h"
#include "binarizeEngine.h"
#include "binarizeGPU.cuh"
#include "../common/logMessage.h"

#define LOG_BINARIZEENGINE_MODULE	"BinarizeEngine"

CBinarizeEngine::CBinarizeEngine()
{
	m_pMoudle = LOG_BINARIZEENGINE_MODULE;
	m_pVolumeData = NULL;
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

bool CBinarizeEngine::binarizeOnHost(SDS3D& stBinarizeData)
{
	bool bRet = false;

	do 
	{
		if (!m_pVolumeData) 
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data is Null!").c_str());
			break;
		}

		int nVolSize = m_pVolumeData->nWid * m_pVolumeData->nHei * m_pVolumeData->nNum;
		int nBinSize = stBinarizeData.nWid * stBinarizeData.nHei * stBinarizeData.nNum;

		if (nVolSize != nBinSize)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data size is different binarize data, %d / %d!" , nVolSize, nBinSize).c_str());
			break;
		}

		m_stTimer.startTimer("Binarize on host");
		for (int z = 0; z < m_pVolumeData->nNum; z++)
		{
			for (int y = 0; y < m_pVolumeData->nHei; y++)
			{
				for (int x = 0; x < m_pVolumeData->nWid; x++)
				{
					int nIdx = x + m_pVolumeData->nWid * y + m_pVolumeData->nWid * m_pVolumeData->nHei * z;

					if (m_pVolumeData->pVolumeData[nIdx] < 20)
					{
						stBinarizeData.pVolumeData[nIdx] = 0;
					}
					else
					{
						stBinarizeData.pVolumeData[nIdx] = 1;
					}
				}
			}
		}
		m_stTimer.stopTimer("Binarize on host");

		bRet = true;

	} while (0);

	return bRet;
}

bool CBinarizeEngine::binarizeOnGPU(SDS3D& stBinarizeData)
{
	bool bRet = false;

	do
	{
		if (!m_pVolumeData)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data is Null!").c_str());
			break;
		}

		int nVolSize = m_pVolumeData->nWid * m_pVolumeData->nHei * m_pVolumeData->nNum;
		int nBinSize = stBinarizeData.nWid * stBinarizeData.nHei * stBinarizeData.nNum;

		if (nVolSize != nBinSize)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data size is different binarize data, %d / %d!", nVolSize, nBinSize).c_str());
			break;
		}

		m_stTimer.startTimer("Binarize on GPU");
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->prepare(m_pVolumeData);
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->run(stBinarizeData);
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->release();
		m_stTimer.stopTimer("Binarize on GPU");

		bRet = true;

	} while (0);

	return bRet;
}