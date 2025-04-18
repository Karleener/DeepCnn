
// DeepCnn2025Dlg.h : fichier d'en-tête
//

#pragma once
#include <vector>
#include <string>
#include "CNetcv.h"


// boîte de dialogue de CDeepCnn2025Dlg
class CDeepCnn2025Dlg : public CDialogEx
{
// Construction
public:
	CDeepCnn2025Dlg(CWnd* pParent = nullptr);	// constructeur standard
	bool initCUDA();

// Données de boîte de dialogue
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DEEPCNN2025_DIALOG };
#endif
	CNetcv MyNet;

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// Prise en charge de DDX/DDV


// Implémentation
protected:
	HICON m_hIcon;

	// Fonctions générées de la table des messages
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	std::vector<std::string> class_names;
	afx_msg void OnBnClickedButtonConfig();
	afx_msg void OnBnClickedButtonTrain();
	afx_msg void OnBnClickedButtonTest();
	afx_msg void OnEnChangeEditBlob();
	int m_Blob;
};
