#include "stdafx.h"
#include "common.h"

#define LOG_COMMON_MODULE	"Common"

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

void Common::initRandData(short* ip, const int size)
{
	time_t tm;
	srand((unsigned) time(&tm));

	for (int i = 0; i < size; i++)
	{
		ip[i] = (short)(rand() & 0xFF) / 2;
	}
}

void Common::campareResult(short* host, short* gpu, const unsigned int size)
{
	if (!host || !gpu)
		return;

	double epsilon = 1.0E-8;
	bool match = true;
	int count = 0;

	for (count = 0; count < size; count++)
	{
		if (fabs(host[count] - gpu[count] > epsilon))
		{
			match = false;
			log_info(LOG_COMMON_MODULE, LogFormatA_A("Host and GPU result do not match!\n").c_str());
			log_info(LOG_COMMON_MODULE, LogFormatA_A("Host %5.2f GPU %5.2f at current %d\n", host[count], gpu[count], count).c_str());
			break;
		}
	}

	if (match == true)
	{
		log_info(LOG_COMMON_MODULE, LogFormatA_A("Host and GPU result match! Data size %d\n", count).c_str());
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
				log_info(LOG_COMMON_MODULE, LogFormatA_A("Host array and GPU array result do not match!\n").c_str());
				log_info(LOG_COMMON_MODULE, LogFormatA_A("Host %5.2f GPU %5.2f at current [%d/%d]\n", hostArray[i][j], gpuArray[i][j], width, height).c_str());
				break;
			}
		}
	}

	if (match == true)
	{
		log_info(LOG_COMMON_MODULE, LogFormatA_A("Host and GPU result match! Data size %d\n", height * width).c_str());
	}
}

int Common::calcDimSize(vdim3 dim)
{
	int nSize = 0;
	nSize = dim.col * dim.row * dim.hei;
	return nSize;
}