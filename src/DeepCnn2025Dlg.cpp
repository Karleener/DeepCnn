
// DeepCnn2025Dlg.cpp : fichier d'implémentation
//
#pragma execution_character_set("utf-8")
#include "pch.h"
#include "framework.h"
#include "DeepCnn2025.h"
#include "DeepCnn2025Dlg.h"
#include "afxdialogex.h"
#include "CNetcv.h"
#include "CNNDialog.h"
#include "CDialTrain.h"
#include "CDialTest.h"
#include "DeviceManager.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif



using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace cv::dnn;




// boîte de dialogue CAboutDlg utilisée pour la boîte de dialogue 'À propos de' pour votre application

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Données de boîte de dialogue
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // Prise en charge de DDX/DDV

// Implémentation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// boîte de dialogue de CDeepCnn2025Dlg



CDeepCnn2025Dlg::CDeepCnn2025Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DEEPCNN2025_DIALOG, pParent)
	, m_Blob(28)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CDeepCnn2025Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_BLOB, m_Blob);
	DDV_MinMaxInt(pDX, m_Blob, 14, 512);
}

BEGIN_MESSAGE_MAP(CDeepCnn2025Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_CONFIG, &CDeepCnn2025Dlg::OnBnClickedButtonConfig)
	ON_BN_CLICKED(IDC_BUTTON_TRAIN, &CDeepCnn2025Dlg::OnBnClickedButtonTrain)
	ON_BN_CLICKED(IDC_BUTTON_TEST, &CDeepCnn2025Dlg::OnBnClickedButtonTest)
	ON_EN_CHANGE(IDC_EDIT_BLOB, &CDeepCnn2025Dlg::OnEnChangeEditBlob)
END_MESSAGE_MAP()


// gestionnaires de messages de CDeepCnn2025Dlg

BOOL CDeepCnn2025Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Ajouter l'élément de menu "À propos de..." au menu Système.

	// IDM_ABOUTBOX doit se trouver dans la plage des commandes système.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Définir l'icône de cette boîte de dialogue.  L'infrastructure effectue cela automatiquement
	//  lorsque la fenêtre principale de l'application n'est pas une boîte de dialogue
	SetIcon(m_hIcon, TRUE);			// Définir une grande icône
	SetIcon(m_hIcon, FALSE);		// Définir une petite icône

	// TODO: ajoutez ici une initialisation supplémentaire
	initCUDA();
	return TRUE;  // retourne TRUE, sauf si vous avez défini le focus sur un contrôle
}

void CDeepCnn2025Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// Si vous ajoutez un bouton Réduire à votre boîte de dialogue, vous devez utiliser le code ci-dessous
//  pour dessiner l'icône.  Pour les applications MFC utilisant le modèle Document/Vue,
//  cela est fait automatiquement par l'infrastructure.

void CDeepCnn2025Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // contexte de périphérique pour la peinture

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Centrer l'icône dans le rectangle client
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Dessiner l'icône
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// Le système appelle cette fonction pour obtenir le curseur à afficher lorsque l'utilisateur fait glisser
//  la fenêtre réduite.
HCURSOR CDeepCnn2025Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CDeepCnn2025Dlg::OnBnClickedButtonConfig()
{
	UpdateData(true);
	CCNNDialog dlg(m_Blob, m_Blob);
	dlg.DoModal();
}

void CDeepCnn2025Dlg::OnBnClickedButtonTrain()
{
	UpdateData(true);
	MyNet.m_input_height = MyNet.m_input_width = m_Blob;
	CDialTrain dlg(NULL,&MyNet);
	dlg.DoModal();


}

void CDeepCnn2025Dlg::OnBnClickedButtonTest()
{
	UpdateData(true);
	CDialTest dlgtest(NULL,&MyNet);
	MyNet.m_input_height = MyNet.m_input_width = m_Blob;
	dlgtest.DoModal();


}

void CDeepCnn2025Dlg::OnEnChangeEditBlob()
{
	// TODO:  S'il s'agit d'un contrôle RICHEDIT, le contrôle ne
	// envoyez cette notification sauf si vous substituez CDialogEx::OnInitDialog()
	// fonction et appelle CRichEditCtrl().SetEventMask()
	// avec l'indicateur ENM_CHANGE ajouté au masque grâce à l'opérateur OR.

	// TODO:  Ajoutez ici le code de votre gestionnaire de notification de contrôle
}


bool CDeepCnn2025Dlg::initCUDA() 
{
	// Charger explicitement la DLL CUDA
	try {
		HMODULE hTorchCuda = LoadLibraryA("torch_cuda.dll");
		if (!hTorchCuda) {
			SetWindowText(L"CPU Only");
			return false;
		}
		else
		{
			SetWindowText(L"GPU OK");
		}
	}
	catch (const std::exception& e) {
		SetWindowText(L"CPU Only");
		return false;
	}

	//if (!hTorchCuda) {
	//	SetWindowText(L"Erreur de chargement de la DLL torch_cuda.dll Le CPU sera utilisé");
	////	MessageBoxA(NULL, "Erreur de chargement de la DLL torch_cuda.dll\n Le CPU sera utilisé", "Erreur", MB_OK | MB_ICONERROR);
	//	return false;
	//}

	//// Vérifier si CUDA est disponible
	//if (!torch::cuda::is_available()) {
	//	SetWindowText(L"Erreur de chargement de la DLL torch_cuda.dll Le CPU sera utilisé");
	//	return false;
	//}


	////MessageBoxA(NULL, "Le GPU est disponible", "Information", MB_OK | MB_ICONINFORMATION);
	//SetWindowText(L"Le GPU est disponible");
	return true;
}