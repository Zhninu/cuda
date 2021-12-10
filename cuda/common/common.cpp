#include "stdafx.h"
#include "common.h"

#define LOG_COMMON_MODULE	"Common"

bool Common::mallocVolume(SDS3D* volume, vdim3 dim)
{
	bool bMalloc = false;
	if (volume)
		return bMalloc;

	volume = new SDS3D;
	volume->dim = dim;

	unsigned long nSize = Common::calcDimSize(volume->dim);
	unsigned long nBytes = nSize * sizeof(short);

	volume->data = (short*)malloc(nBytes);
	Common::initRandData(volume->data, nSize);
	bMalloc = true;

	return bMalloc;
}

bool Common::freeVolume(SDS3D* volume)
{	
	bool bFree = false;

	if (volume)
		return bFree;

	if (volume->data)
	{
		free(volume->data);
		volume->data = NULL;
	}

	delete volume;
	volume = NULL;
	bFree = true;

	return bFree;
}

void Common::allocArray2D(short** array2D, vdim3 dim)
{
	unsigned int height = dim.hei;
	unsigned int width = dim.col * dim.row;

	array2D = new short* [dim.hei];
	for (unsigned int i = 0; i < height; i++)
	{
		array2D[i] = new short [width];
	}
}

void Common::freeArray2D(short** array2D, vdim3 dim)
{
	unsigned int height = dim.hei;
	unsigned int width = dim.col * dim.row;

	for (unsigned int i = 0; i < height; i++)
	{
		delete array2D[i];
	}
	delete []array2D;
}

void Common::initArray2D(short** array2D, vdim3 dim)
{
	time_t tm;
	srand((unsigned)time(&tm));

	unsigned int height = dim.hei;
	unsigned int width = dim.col * dim.row;

	for (unsigned int i = 0; i < height; i++)
	{
		for (unsigned int j = 0; j < width; j++) 
		{
			array2D[i][j] = (short)(rand() & 0xFF) / 2;
		}
	}
}

void Common::convertArray2D(SDS1D* array1D, SDS2D* array2D, cvArray type)
{
	if (!array1D || !array2D)
		return;

	if (array1D->size != calcDimSize(array2D->dim))
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

void Common::initRandData(short* ip, const unsigned long size)
{
	time_t tm;
	srand((unsigned) time(&tm));

	for (unsigned long i = 0; i < size; i++)
	{
		ip[i] = (short)(rand() & 0xFF) / 2;
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

unsigned long Common::calcDimSize(vdim3 dim)
{
	unsigned long nSize = 0;
	nSize = (unsigned long)dim.col * dim.row * dim.hei;
	return nSize;
}

unsigned long Common::calcDimSize(vdim2 dim)
{
	unsigned long nSize = 0;
	nSize = (unsigned long)dim.col * dim.row;
	return nSize;
}