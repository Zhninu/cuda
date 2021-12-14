#pragma once
#include "stdafx.h"
#include "binarize_engine.h"
#include "binarize_gpu.cuh"
#include "../common/common.h"
#include "../common/log_message.h"

CBinarizeEngine::CBinarizeEngine(SDS3D* volumedata, int thresh, int maxval)
	: m_pMoudle(LOG_BINARIZE_ENGINE_MODULE)
	, m_pclsBinarize(NULL)
	, m_pVolumeData(volumedata)
	, m_nThresh(thresh)
	, m_nMaxVal(maxval)
	, m_bCreateVol(false)
{
	if(!m_pclsBinarize)
		m_pclsBinarize = new CBinarizeGPU;

	if (!m_pVolumeData)
	{
		vdim3 voldim(BINARY_VOLUME_COLUME, BINARY_VOLUME_ROW, BINARY_VOLUME_HEIGHT);
		m_bCreateVol = Common::mallocVolume(&m_pVolumeData, voldim);
	}
}

CBinarizeEngine::CBinarizeEngine(vdim3 dim, int thresh, int maxval)
	: m_pMoudle(LOG_BINARIZE_ENGINE_MODULE)
	, m_pclsBinarize(NULL)
	, m_pVolumeData(NULL)
	, m_nThresh(thresh)
	, m_nMaxVal(maxval)
	, m_bCreateVol(false)
{
	if (!m_pclsBinarize)
		m_pclsBinarize = new CBinarizeGPU;

	if (!m_pVolumeData)
		m_bCreateVol = Common::mallocVolume(&m_pVolumeData, dim);
}

CBinarizeEngine::~CBinarizeEngine()
{
	if(m_pclsBinarize)
		delete(reinterpret_cast<CBinarizeGPU *>(m_pclsBinarize));

	if (m_bCreateVol) 
	{
		m_bCreateVol = Common::freeVolume(&m_pVolumeData);
	}
}


int CBinarizeEngine::binarize()
{
	int	  nErr = EC_OK;

	binSDS3D binaryVol;
	binaryVol.dim = m_pVolumeData->dim;
	binaryVol.thresh = m_nThresh;
	binaryVol.maxval = m_nMaxVal;

	binSDS3D binaryVolGPU(binaryVol);

	unsigned long nSize = Common::calcDim(binaryVol.dim);
	unsigned long nBytes = nSize * sizeof(short);
	binaryVol.data = (short*)malloc(nBytes);
	binaryVolGPU.data = (short*)malloc(nBytes);

	log_info(m_pMoudle, LogFormatA_A("Binarize starting! Volume dimension %d/%d/%d", 
										m_pVolumeData->dim.col, m_pVolumeData->dim.row, m_pVolumeData->dim.hei).c_str());

	//start binary
	binarizeHost(binaryVol);
	binarizeDev(binaryVolGPU);

	//check result
	Common::campareResult(binaryVol.data, binaryVolGPU.data, nSize);

	free(binaryVol.data);
	free(binaryVolGPU.data);

	return nErr;
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

		int nThresh = binarydata.thresh;
		int nMaxVal = binarydata.maxval;
		unsigned long nVolSize = Common::calcDim(m_pVolumeData->dim);
		unsigned long nBinSize = Common::calcDim(binarydata.dim);
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
					unsigned long nIdx = x + m_pVolumeData->dim.col * y + m_pVolumeData->dim.col * m_pVolumeData->dim.row * z;
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

		unsigned long nVolSize = Common::calcDim(m_pVolumeData->dim);
		unsigned long nBinSize = Common::calcDim(binarydata.dim);
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