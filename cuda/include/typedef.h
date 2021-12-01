#pragma once
#ifndef _TYPRDEF_H_
#define _TYPRDEF_H_

#define EC_OK	0
#define EC_ERR	1

typedef enum tagTimerType
{
	timer_none = 0,
	timer_time,				//C系统调用, <1s
	timer_clock,			//C系统调用, <10ms
	timer_ApiTime,			//Windows API, <1ms, timeGetTime
	timer_QPCounter,		//Windows API, <<0.1ms, QueryPerformanceCounter
	timer_TickCount,		//CWindows API, <1ms, GetTickCount
}timer;

typedef enum tagLogLevel
{
	level_debug = 0,
	level_info,
	level_warning,
	level_error,
}LogLevel;


typedef struct tagSDS3D
{
	short*	pVolumeData;
	short	nHei;
	short	nWid;
	short	nNum;
}SDS3D;

#endif // _TYPRDEF_H_