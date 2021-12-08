#pragma once
#include "stdafx.h"
#include "binarize_engine.h"
#include "binarize_gpu.cuh"
#include "../common/common.h"
#include "../common/log_message.h"

#define LOG_BINARIZEENGINE_MODULE	"BinarizeEngine"

CBinarizeEngine::CBinarizeEngine(SDS3D* volumedata, int thresh, int maxval)
	: m_pMoudle(LOG_BINARIZEENGINE_MODULE)
	, m_pVolumeData(volumedata)
	, m_nThresh(thresh)
	, m_nMaxVal(maxval)
	, m_bCreateVol(false)
{
	m_pclsBinarize = new CBinarizeGPU;
	if (!m_pVolumeData)
	{
		vdim3 voldim(VOLUME_COLUME, VOLUME_ROW, VOLUME_HEIGHT);
		createVolume(voldim);
	}
}

CBinarizeEngine::CBinarizeEngine(vdim3 dim, int thresh, int maxval)
	: m_pMoudle(LOG_BINARIZEENGINE_MODULE)
	, m_pVolumeData(NULL)
	, m_nThresh(thresh)
	, m_nMaxVal(maxval)
	, m_bCreateVol(false)
{
	createVolume(dim);
}

CBinarizeEngine::~CBinarizeEngine()
{
	delete(reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize));
	freeVolume();
}


int CBinarizeEngine::binarize()
{
	int	  nErr = EC_OK;

	binSDS3D binaryVol;
	binaryVol.dim = m_pVolumeData->dim;
	binaryVol.thresh = m_nThresh;
	binaryVol.maxval = m_nMaxVal;
	binSDS3D binaryVolGPU(binaryVol);

	int nSize = calcDimSize(m_pVolumeData->dim);
	int nBytes = nSize * sizeof(short);

	binaryVol.data = (short*)malloc(nBytes);
	binaryVolGPU.data = (short*)malloc(nBytes);

	//Start binary
	binarizeHost(binaryVol);
	binarizeDev(binaryVolGPU);

	campareResult(binaryVol.data, binaryVolGPU.data, nSize);

	free(binaryVol.data);
	free(binaryVolGPU.data);

	return nErr;
}

void CBinarizeEngine::createVolume(vdim3 dim)
{
	if (m_pVolumeData)
		return;

	m_pVolumeData = new SDS3D;
	m_pVolumeData->dim = dim;

	int nSize = calcDimSize(m_pVolumeData->dim);
	int nBytes = nSize * sizeof(short);

	m_pVolumeData->data = (short*)malloc(nBytes);
	initRandData(m_pVolumeData->data, nSize);

	m_bCreateVol = true;
}

void CBinarizeEngine::freeVolume()
{
	if (m_bCreateVol && m_pVolumeData) 
	{
		if (m_pVolumeData->data) 
		{
			free(m_pVolumeData->data);
			m_pVolumeData->data = NULL;
		}

		free(m_pVolumeData);
		m_pVolumeData = NULL;
		m_bCreateVol = false;
	}
}

bool CBinarizeEngine::binarizeHost(binSDS3D& binarydata)
{
	bool bRet = false;

	do
	{
		if (!m_pVolumeData)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data is Null!").c_str());
			break;
		}

		int nVolSize = calcDimSize(m_pVolumeData->dim);
		int nBinSize = calcDimSize(binarydata.dim);
		int nThresh = binarydata.thresh;
		int nMaxVal = binarydata.maxval;

		if (nVolSize != nBinSize)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data size is different binarize data, %d / %d!", nVolSize, nBinSize).c_str());
			break;
		}

		m_stTimer.startTimer("Binarize on host");
		for (unsigned int z = 0; z < m_pVolumeData->dim.hei; z++)
		{
			for (unsigned int y = 0; y < m_pVolumeData->dim.row; y++)
			{
				for (unsigned int x = 0; x < m_pVolumeData->dim.col; x++)
				{
					int nIdx = x + m_pVolumeData->dim.col * y + m_pVolumeData->dim.col * m_pVolumeData->dim.row * z;

					if (m_pVolumeData->data[nIdx] < nThresh)
					{
						binarydata.data[nIdx] = m_pVolumeData->data[nIdx];
					}
					else
					{
						binarydata.data[nIdx] = nMaxVal;
					}
				}
			}
		}
		m_stTimer.stopTimer("Binarize on host");

		bRet = true;

	} while (0);

	return bRet;
}

bool CBinarizeEngine::binarizeDev(binSDS3D& binarydata)
{
	bool bRet = false;

	do
	{
		if (!m_pVolumeData)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data is Null!").c_str());
			break;
		}

		int nVolSize = calcDimSize(m_pVolumeData->dim);
		int nBinSize = calcDimSize(binarydata.dim);

		if (nVolSize != nBinSize)
		{
			log_error(m_pMoudle, LogFormatA_A("Volume data size is different binarize data, %d / %d!", nVolSize, nBinSize).c_str());
			break;
		}

		m_stTimer.startTimer("Binarize on GPU");
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->prepare(m_pVolumeData);
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->run(binarydata);
		reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize)->release();
		m_stTimer.stopTimer("Binarize on GPU");

		bRet = true;

	} while (0);

	return bRet;
}