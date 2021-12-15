#include "stdafx.h"
#include "common.h"

#define LOG_COMMON_MODULE	"Common"

Module Common::findMode(char* str) 
{
	Module nMType = Module_All;

	do 
	{
		if (!str)
			break;

		if (stricmp(str, "binary") == 0) 
		{
			nMType = Module_Binary;
			break;
		}

		if (stricmp(str, "interp") == 0)
		{
			nMType = Module_Interpolation;
			break;
		}

		if (stricmp(str, "cudaverify") == 0)
		{
			nMType = Module_CudaVerify;
			break;
		}

	} while (0);

	return nMType;
}

void Common::convertArray(SDS1D* a1d, SDS2D* a2d, cvArray type)
{
	SDS1D* array1D = a1d;
	SDS2D* array2D = a2d;

	if (!array1D || !array2D)
		return;

	if (array1D->size != calcDim(array2D->dim))
		return;

	if (type != cvArray_1DTo2D || type != cvArray_2DTo1D)
		return;

	unsigned int size = array1D->size;
	unsigned int width = array2D->dim.col;
	unsigned int height = array2D->dim.row;

	for (unsigned int row = 0; row < height; row++)
	{
		for (unsigned int col = 0; col < width; col++)
		{
			unsigned int uIdx = col + row * width;
			if (type == cvArray_1DTo2D) 
			{
				array2D->data[row][col] = array1D->data[uIdx];
			}
			else if(type == cvArray_2DTo1D)
			{
				array1D->data[uIdx] = array2D->data[row][col];
			}
		}
	}
}


void Common::campareResult(short* host, short* gpu, const unsigned long size)
{
	if (!host || !gpu)
		return;

	double epsilon = 1.0E-8;
	bool match = true;
	unsigned long count = 0;

	for (count = 0; count < size; count++)
	{
		if (fabs(host[count] - gpu[count] > epsilon))
		{
			match = false;
			log_info(LOG_COMMON_MODULE, LogFormatA_A("Host and GPU result do not match!").c_str());
			log_info(LOG_COMMON_MODULE, LogFormatA_A("Host %5.2f GPU %5.2f at current %d", host[count], gpu[count], count).c_str());
			break;
		}
	}

	if (match == true)
	{
		log_info(LOG_COMMON_MODULE, LogFormatA_A("Host and GPU result match! Data size %d", count).c_str());
	}
}

void Common::campareResult(short** hostArray, short** gpuArray, const vdim3 dim)
{
	if (!hostArray || !gpuArray)
		return;

	double epsilon = 1.0E-8;
	bool match = true;
	unsigned int height = dim.hei;
	unsigned int width = dim.col * dim.row;

	for (unsigned int i = 0; i < height; i++)
	{
		for (unsigned int j = 0; j < width; j++)
		{
			if (fabs(hostArray[i][j] - gpuArray[i][j] > epsilon))
			{
				match = false;
				log_info(LOG_COMMON_MODULE, LogFormatA_A("Host array and GPU array result do not match!").c_str());
				log_info(LOG_COMMON_MODULE, LogFormatA_A("Host %5.2f GPU %5.2f at current [%d/%d]", hostArray[i][j], gpuArray[i][j], width, height).c_str());
				break;
			}
		}
	}

	if (match == true)
	{
		log_info(LOG_COMMON_MODULE, LogFormatA_A("Host and GPU result match! Data size %d", height * width).c_str());
	}
}

bool Common::campareResult(int *ptr, const int n, const int x)
{
	if (!ptr)
		return false;

	for (int i = 0; i < n; i++)
		if (ptr[i] != x)
		{
			log_info(LOG_COMMON_MODULE, LogFormatA_A("Error! data[%d] = %d, ref = %d\n", i, ptr[i], x).c_str());
			return false;
		}

	return true;
}

unsigned long Common::calcDim(vdim3 dim)
{
	unsigned long nSize = 0;
	nSize = (unsigned long)dim.col * dim.row * dim.hei;
	return nSize;
}

unsigned long Common::calcDim(vdim2 dim)
{
	unsigned long nSize = 0;
	nSize = (unsigned long)dim.col * dim.row;
	return nSize;
}