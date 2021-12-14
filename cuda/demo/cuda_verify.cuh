#include "typedef.h"
#include "../common/timer.h"

class CCudaVerify
{
public:
	CCudaVerify();
	~CCudaVerify();

public:
	int memcpyAsync(int argc, char **argv);

private:
	const char*   m_pMoudle;
	CTimer		  m_stTimer;
};