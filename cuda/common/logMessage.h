#pragma once
#include <string>
#include "typedef.h"

#define log_debug		CLogMsg::getLogMsg()->debug
#define log_info		CLogMsg::getLogMsg()->info
#define log_warning		CLogMsg::getLogMsg()->warning
#define log_error		CLogMsg::getLogMsg()->error

class CLogMsg 
{
public:
	CLogMsg();
	~CLogMsg();

public:
	static CLogMsg* getLogMsg();
	void setLevel(LogLevel level);

	void debug(const char* pModuleName, const char* pMsg, const char* pProcName = NULL, const char* pLogTime = NULL);
	void info(const char* pModuleName, const char* pMsg, const char* pProcName = NULL, const char* pLogTime = NULL);
	void warning(const char* pModuleName, const char* pMsg, const char* pProcName = NULL, const char* pLogTime = NULL);
	void error(const char* pModuleName, const char* pMsg, const char* pProcName = NULL, const char* pLogTime = NULL);

private:
	static CLogMsg*		m_pInstance;
	LogLevel			m_level;
};

//************************************
// Method:    LogFormatA_A
// FullName:  LogFormatA_A
// Access:    public 
// Returns:   std::string
// Qualifier:
// Parameter: const char * pFormat
// Parameter: 必须是char类型的可变参数
//************************************
inline std::string LogFormatA_A(const char* pFormat, ...)
{
	va_list args;
	va_start(args, pFormat);
	char pRet[1024] = { 0 };
	_vsnprintf_s(pRet, 1023, pFormat, args);
	va_end(args);

	return std::string(pRet);
}