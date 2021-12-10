#pragma once
#include "stdafx.h"
#include "timer.h"
#include "log_message.h"

#define  LOG_TIMER_MOUDLE "Timer"

CTimer::CTimer() 
	: m_pMoudle(LOG_TIMER_MOUDLE)
	, m_timer(timer_QPCounter)
	, m_recTime(0)
	, m_recClock(0)
	, m_recApiTime(0)
	,m_recTickCount(0)
{
	memset(&m_recQPCounter, 0, sizeof(LARGE_INTEGER));
}

CTimer::CTimer(timer type)
	: m_pMoudle(LOG_TIMER_MOUDLE)
	, m_recTime(0)
	, m_recClock(0)
	, m_recApiTime(0)
	, m_recTickCount(0)
{
	m_timer = type;
	memset(&m_recQPCounter, 0, sizeof(LARGE_INTEGER));
}

CTimer::~CTimer()
{
}

void CTimer::setTimer(timer type) 
{
	m_timer = type;
}

void CTimer::startTimer(const char* str)
{
	bool bRet = false;

	switch (m_timer)
	{
	case timer_time:
	{
		if(m_recTime)
			break;

		m_recTime = time(NULL);
		bRet = true;
		break;
	}
	case timer_clock:
	{
		if (m_recClock)
			break;

		m_recClock = clock();
		bRet = true;
		break;
	}
	case timer_ApiTime:
	{
		if (m_recApiTime)
			break;

		m_recApiTime = timeGetTime();
		bRet = true;
		break;
	}
	case timer_QPCounter:
	{
		if(m_recQPCounter.QuadPart)
			break;

		QueryPerformanceCounter(&m_recQPCounter);
		bRet = true;
		break;
	}
	case timer_TickCount:
	{
		if(m_recTickCount)
			break;

		m_recTickCount = GetTickCount();
		bRet = true;
		break;
	}
	default:
		break;
	}

	if (bRet) 
	{
		log_info(m_pMoudle, LogFormatA_A("%s timer start, timer:%d", str, m_timer).c_str());
	}
	else 
	{
		log_info(m_pMoudle, LogFormatA_A("%s timer start failed, timer not reset, timer:%d", str, m_timer).c_str());
	}

	return;
}

void CTimer::stopTimer(const char* str)
{
	switch (m_timer)
	{
	case timer_time:
	{
		time_t stopTime;
		stopTime = time(NULL);
		log_info(m_pMoudle, LogFormatA_A("%s timer stop, cost:%lld s[%lld - %lld], timer:%d", str, (stopTime - m_recTime), m_recTime, stopTime, m_timer).c_str());
		m_recTime = 0;
		break;
	}
	case timer_clock:
	{
		clock_t stopClock;
		stopClock = clock();
		log_info(m_pMoudle, LogFormatA_A("%s timer stop, cost:%d ms[%ld - %ld], timer:%d", str, (stopClock - m_recClock), stopClock, m_recClock, m_timer).c_str());
		m_recClock = 0;
		break;
	}
	case timer_ApiTime:
	{
		DWORD stopApiTime;
		stopApiTime = timeGetTime();
		log_info(m_pMoudle, LogFormatA_A("%s timer stop, cost:%d ms[%lu - %lu], timer:%d", str, (stopApiTime - m_recApiTime), stopApiTime, m_recApiTime, m_timer).c_str());
		m_recApiTime = 0;
		break;
	}
	case timer_QPCounter:
	{
		LARGE_INTEGER stopQPCounter, cQPCounter;
		QueryPerformanceFrequency(&cQPCounter);
		QueryPerformanceCounter(&stopQPCounter);
		double cost = (stopQPCounter.QuadPart - m_recQPCounter.QuadPart) * 1.0 / cQPCounter.QuadPart * 1000;
		log_info(m_pMoudle, LogFormatA_A("%s timer stop, cost:%f ms[%lld - %lld], timer:%d", str, cost, stopQPCounter.QuadPart, m_recQPCounter.QuadPart, m_timer).c_str());
		memset(&m_recQPCounter, 0, sizeof(LARGE_INTEGER));
		break;
	}
	case timer_TickCount:
	{
		DWORD stopTickCount;
		stopTickCount = GetTickCount();
		log_info(m_pMoudle, LogFormatA_A("%s timer stop, cost:%d ms[%lu - %lu], timer:%d", str, (stopTickCount - m_recTickCount), stopTickCount, m_recTickCount, m_timer).c_str());
		m_recTickCount = 0;
		break;
	}
	default:
		break;
	}

	return;
}

void CTimer::startTime(time_t& tm, const char* str)
{
	tm = time(NULL);
}

void CTimer::stopTime(time_t tm, const char* str) 
{
	time_t stopTime;
	stopTime = time(NULL);
	log_info(m_pMoudle, LogFormatA_A("[Time] %s timer stop, cost:%lld s[%lld - %lld]", str, (stopTime - tm), tm, stopTime).c_str());
}

void CTimer::startClock(clock_t& clk, const char* str) 
{
	clk = clock();
}

void CTimer::stopClock(clock_t clk, const char* str) 
{
	clock_t stopClock;
	stopClock = clock();
	log_info(m_pMoudle, LogFormatA_A("[Clock] %s timer stop, cost:%ld ms[%ld - %ld]", str, (stopClock - clk), clk, stopClock).c_str());
}

void CTimer::startApiTime(DWORD& tm, const char* str) 
{
	tm = timeGetTime();
}

void CTimer::stopApiTime(DWORD tm, const char* str) 
{
	DWORD stopTime;
	stopTime = timeGetTime();
	log_info(m_pMoudle, LogFormatA_A("[ApiTime] %s timer stop, cost:%lu ms[%lu - %lu]", str, (stopTime - tm), tm, stopTime).c_str());
}

void CTimer::startQPCounter(LARGE_INTEGER& counter, const char* str) 
{
	QueryPerformanceCounter(&counter);
}

void CTimer::stopQPCounter(LARGE_INTEGER counter, const char* str) 
{
	LARGE_INTEGER stopQPCounter, cQPCounter;
	QueryPerformanceFrequency(&cQPCounter);
	QueryPerformanceCounter(&stopQPCounter);
	double cost = (stopQPCounter.QuadPart - counter.QuadPart) * 1.0 / cQPCounter.QuadPart * 1000;
	log_info(m_pMoudle, LogFormatA_A("[QPCounter] %s timer stop, cost:%f ms[%lld - %lld]", str, cost, counter.QuadPart, stopQPCounter.QuadPart).c_str());
}

void CTimer::startTickCount(DWORD& tick, const char* str) 
{
	tick = GetTickCount();
}

void CTimer::stopTickCount(DWORD tick, const char* str) 
{
	DWORD stopTick;
	stopTick = GetTickCount();
	log_info(m_pMoudle, LogFormatA_A("[Tick] %s timer stop, cost:%lu ms[%lu - %lu]", str, (stopTick - tick), tick, stopTick).c_str());
}