#include "stdafx.h"
#include "common.h"

#define LOG_COMMON_MODULE	"Common"

void initRandData(short* ip, const int nSize) 
{
	time_t tm;
	srand((unsigned) time(&tm));

	for (int i = 0; i < nSize; i++)
	{
		ip[i] = (short)(rand() & 0xFF) / 2;
	}
}

void campareResult(short* host, short* gpu, const int nSize)
{
	if (!host || !gpu)
		return;

	double epsilon = 1.0E-8;
	bool match = true;
	int count = 0;
	for (count = 0; count < nSize; count++)
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

int calcDimSize(VolDim dim)
{
	int nSize = 0;
	nSize = dim.nCol * dim.nRow * dim.nHei;
	return nSize;
}