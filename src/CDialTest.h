#pragma once
#include "afxdialogex.h"
#include <string>
#include <vector>
#include "CNetcv.h"
#include "CNNDialog.h"

// boîte de dialogue de CDialTest

class CDialTest : public CDialogEx
{
	DECLARE_DYNAMIC(CDialTest)

public:
	CDialTest(CWnd* pParent = nullptr, CNetcv* pMyNet=nullptr);   // constructeur standard
	virtual ~CDialTest();
	CNetcv *m_pMyNet;
// Données de boîte de dialogue
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_TEST };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // Prise en charge de DDX/DDV

	DECLARE_MESSAGE_MAP()
public:
	static void onMouse(int event, int x, int y, int flags, void* userdata);


	String filePath;
	String m_image_path;
	afx_msg void OnBnClickedButtonChoixmodel();
	afx_msg void OnBnClickedButtonChoiximage();
	CString m_Texte;
	afx_msg void OnBnClickedButtonChoiximageSeg();
	afx_msg void OnBnClickedButtonChoiximageGt();

	cv::Mat m_currentImage;
	std::string m_outputDirectory;
	int m_currentClass;
	int m_patchSize;
	afx_msg void OnBnClickedButtonCreateTrainTest();
	int m_blob;
	virtual BOOL OnInitDialog();
	int m_Tol;
	BOOL m_boolEdges;
};
