#pragma once
#include "afxdialogex.h"
#include <string>
#include <vector>
#include "CNetcv.h"


// boîte de dialogue de CDialTrain

class CDialTrain : public CDialogEx
{
	DECLARE_DYNAMIC(CDialTrain)

public:
	CDialTrain(CWnd* pParent = nullptr, CNetcv* pMyNet = nullptr);   // constructeur standard
	virtual ~CDialTrain();
	CNetcv* m_pMyNet; // Utilisation d'un pointeur

#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_TRAIN };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // Prise en charge de DDX/DDV

	DECLARE_MESSAGE_MAP()
public:
	std::vector<std::string> class_names;
	double m_LearningRate;
	int m_Iter;
	CString m_NomModel;
	afx_msg void OnBnClickedButtonLancer();
	int m_Batch;
	CString m_Texte;
	CButton m_BtLancer;
	CButton m_BtOk;
	CString m_Texte_Train;
	int m_Periode;
	bool m_FromScratch;
	afx_msg void OnBnClickedButtonLancerFromModel();
	afx_msg void OnBnClickedButtonGenerepython();
	BOOL m_L2reg;
	BOOL m_AutomLR;
};
