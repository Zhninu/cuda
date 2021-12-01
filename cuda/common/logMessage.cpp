#include "stdafx.h"
#include "logMessage.h"

CLogMsg* CLogMsg::m_pInstance = NULL;

CLogMsg* CLogMsg::getLogMsg()
{
	if (!m_pInstance)
	{
		m_pInstance = new CLogMsg;
	}
	return m_pInstance;
}

CLogMsg::CLogMsg() : m_level(level_error)
{

}

CLogMsg::~CLogMsg()
{

}

void  CLogMsg::setLevel(LogLevel level)
{
	m_level = level;
}
	
void CLogMsg::debug(const char* pModuleName, const char* pMsg, const char* pProcName, const char* pLogTime)
{
	if (m_level >= level_debug) 
	{
		printf("[Debug] Module:%s, %s\n", pModuleName, pMsg);
	}
}

void CLogMsg::info(const char* pModuleName, const char* pMsg, const char* pProcName, const char* pLogTime)
{
	if (m_level >= level_info)
	{
		printf("[Info] Module:%s, %s\n", pModuleName, pMsg);
	}
}

void CLogMsg::warning(const char* pModuleName, const char* pMsg, const char* pProcName, const char* pLogTime)
{
	if (m_level >= level_warning)
	{
		printf("[Warning] Module:%s, %s\n", pModuleName, pMsg);
	}
}

void CLogMsg::error(const char* pModuleName, const char* pMsg, const char* pProcName, const char* pLogTime)
{
	if (m_level >= level_error)
	{
		printf("[Error] Module:%s, %s\n", pModuleName, pMsg);
	}
}