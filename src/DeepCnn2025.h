
// DeepCnn2025.h : fichier d'en-tête principal de l'application PROJECT_NAME
//

#pragma once

#ifndef __AFXWIN_H__
	#error "incluez 'pch.h' avant d'inclure ce fichier pour PCH"
#endif

#include "resource.h"		// symboles principaux


// CDeepCnn2025App :
// Consultez DeepCnn2025.cpp pour l'implémentation de cette classe
//

class CDeepCnn2025App : public CWinApp
{
public:
	CDeepCnn2025App();

// Substitutions
public:
	virtual BOOL InitInstance();

// Implémentation

	DECLARE_MESSAGE_MAP()
};

extern CDeepCnn2025App theApp;
