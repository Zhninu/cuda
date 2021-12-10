#pragma once
#include "typedef.h"

class CTimer
{
public:
	CTimer();
	CTimer(timer type);
	~CTimer();

public:
	void setTimer(timer type);
	void startTimer(const char* str);
	void stopTimer(const char* str);
	void startTime(time_t& tm, const char* str);
	void stopTime(time_t tm, const char* str);
	void startClock(clock_t& clk, const char* str);
	void stopClock(clock_t clk, const char* str);
	void startApiTime(DWORD& tm, const char* str);
	void stopApiTime(DWORD tm, const char* str);
	void startQPCounter(LARGE_INTEGER& counter, const char* str);
	void stopQPCounter(LARGE_INTEGER counter, const char* str);
	void startTickCount(DWORD& tick, const char* str);
	void stopTickCount(DWORD tick, const char* str);

private:
	const char*     m_pMoudle;
	timer			m_timer;
	time_t			m_recTime;
	clock_t			m_recClock;
	DWORD			m_recApiTime;
	LARGE_INTEGER	m_recQPCounter;
	DWORD			m_recTickCount;
};
