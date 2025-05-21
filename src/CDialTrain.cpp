// CDialTrain.cpp : fichier d'implémentation
//

#include "pch.h"
#include "DeepCnn2025.h"
#include "afxdialogex.h"
#include "CDialTrain.h"
#include "CMistral.h"

#include "CNetcv.h"
#include "CNNDialog.h"

// boîte de dialogue de CDialTrain

IMPLEMENT_DYNAMIC(CDialTrain, CDialogEx)

CDialTrain::CDialTrain(CWnd* pParent, CNetcv* pMyNet)
	: CDialogEx(IDD_DIALOG_TRAIN, pParent), m_pMyNet(pMyNet)
	, m_LearningRate(0.001)
	, m_Iter(50)
	, m_NomModel(_T("model_cnn"))
	, m_Batch(5)
	, m_Texte(_T(""))
	, m_Texte_Train(_T(""))
	, m_Periode(50)
	, m_FromScratch(true)
	, m_L2reg(true)
	, m_AutomLR(true)
{

}


CDialTrain::~CDialTrain()
{
}

void CDialTrain::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_LR, m_LearningRate);
	DDX_Text(pDX, IDC_EDIT_ITER, m_Iter);
	DDX_Text(pDX, IDC_EDIT_NOMMODEL, m_NomModel);
	DDX_Text(pDX, IDC_EDIT_BATCH, m_Batch);
	DDV_MinMaxInt(pDX, m_Batch, 1, 1000);
	DDX_Control(pDX, IDC_BUTTON_LANCER, m_BtLancer);
	DDX_Control(pDX, IDOK, m_BtOk);
	DDX_Text(pDX, IDC_STATIC_TEXTE_TRAIN, m_Texte_Train);
	DDX_Text(pDX, IDC_EDIT_SAUVEN, m_Periode);
	DDV_MinMaxInt(pDX, m_Periode, 1, 10000);
	DDX_Check(pDX, IDC_CHECK_L2, m_L2reg);
	DDX_Check(pDX, IDC_CHECK_ADAPTLR, m_AutomLR);
}


BEGIN_MESSAGE_MAP(CDialTrain, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON_LANCER, &CDialTrain::OnBnClickedButtonLancer)
	ON_BN_CLICKED(IDC_BUTTON_LANCER_FROM_MODEL, &CDialTrain::OnBnClickedButtonLancerFromModel)
	ON_BN_CLICKED(IDC_BUTTON_GENEREPYTHON, &CDialTrain::OnBnClickedButtonGenerepython)
END_MESSAGE_MAP()



// gestionnaires de messages de CDialTrain

void CDialTrain::OnBnClickedButtonLancer()
{
	// TODO: ajoutez ici le code de votre gestionnaire de commande
	UpdateData(TRUE);
	m_pMyNet->m_AutomLR = m_AutomLR;
	m_pMyNet->m_L2reg = m_L2reg;
	//int input_height = 28;
	//int input_width = 28;
	m_Texte_Train = L"Apprentissage en cours... \nF2 pour arrêter";
	UpdateData(false);


	// Désactiver les boutons
	m_BtLancer.EnableWindow(FALSE);
	m_BtOk.EnableWindow(FALSE);


	CFileDialog fileDialog(TRUE, _T("configCNN"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Config Files (*.configCNN)|*.configCNN||"));
	fileDialog.m_ofn.lpstrTitle = _T("Charger la configuration");

	if (fileDialog.DoModal() != IDOK)
	{
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);

		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}
	CString configFilePath = fileDialog.GetPathName();
	std::string stdconfigFilePath = CT2A(configFilePath.GetString());
	std::filesystem::path configPath(configFilePath.GetString());
	std::filesystem::path configDir = configPath.parent_path();

	// Charger la configuration à partir du fichier
	std::vector<ConvLayerParams> layers;
	std::vector<DenseLayerParams> Denselayers;
	if (!m_pMyNet->loadConfig(stdconfigFilePath, layers, Denselayers))
	{
		AfxMessageBox(_T("Erreur lors du chargement de la configuration."));
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}
	if (layers.size() == 0)
	{
		AfxMessageBox(_T("Erreur: aucune couche de convolution trouvée dans le fichier de configuration."));
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}
	// Afficher la boîte de dialogue CCNNDialog sous forme de boîte non modale
	CCNNDialog* pDialog = new CCNNDialog(m_pMyNet->m_input_height, m_pMyNet->m_input_width);
	pDialog->Create(IDD_CNN_DIALOG, this);
	pDialog->ShowWindow(SW_SHOW);

	// Reconstruire la liste à partir du fichier de configuration
	pDialog->SetLayers(layers,Denselayers);
	pDialog->m_Visu.SetLayers(layers, Denselayers);
	pDialog->UpdateTotalParams();

	//désactiver les boutons de la boîte de dialogue pDialog
	pDialog->m_BtConv.EnableWindow(FALSE);
	pDialog->m_BtDense.EnableWindow(FALSE);
	pDialog->m_BtSave.EnableWindow(FALSE);


	MessageBoxA(NULL, "Selectionnez le dossier qui contient les sous-dossiers train et test", "Selectionner le dossier", MB_OK);

	CFolderPickerDialog folderPickerDialog(NULL, OFN_FILEMUSTEXIST, this, 0);
	if (folderPickerDialog.DoModal() != IDOK) {
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}

	CString folderPath = folderPickerDialog.GetPathName();
	std::filesystem::path rootPath(folderPath.GetString());

	int NbClasses;

	bool read = m_pMyNet->loadDataset(rootPath.string()); // fixe le nombre de classes et les tensors
	if (!read)
	{
		AfxMessageBox(_T("Directory structure should be	\ntrain\n  classe1\n   image1.jpg\n   image2.jpg\n  classe2\n   image1.jpg\n...\n\ntest\n  classe1\n   image1.jpg\n   image2.jpg\netc."));
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Bad Directory structure";
		UpdateData(false);
		pDialog->DestroyWindow();

		return;
	}

	std::filesystem::path modelFilePath = configDir / std::string(CT2A(m_NomModel));
	std::filesystem::path namesFilePath = configDir / (modelFilePath.stem().string() + ".names");

	if (m_Batch > m_pMyNet->m_train_image_paths.size()) m_Batch = m_pMyNet->m_train_image_paths.size();
	if (m_FromScratch)
	{
		m_pMyNet->m_model = m_pMyNet->createModel(layers, Denselayers, m_pMyNet->m_input_height, m_pMyNet->m_input_width, m_pMyNet->m_nb_classes);
	}
	else
	{ // choisir le modèle existant avec cfiledialog
		CFileDialog fileDialog(TRUE, _T("model_cnn.pt"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Model Files (*.pt)|*.pt||"));
		fileDialog.m_ofn.lpstrTitle = _T("Charger le modèle");
		if (fileDialog.DoModal() != IDOK)
		{
			m_BtLancer.EnableWindow(true);
			m_BtOk.EnableWindow(true);
			m_Texte_Train = L"Opération annulée";
			UpdateData(false);
			pDialog->DestroyWindow();
			return;
		}
		string modelfile = CT2A(fileDialog.GetPathName());
		std::string modelName = modelfile.substr(0, modelfile.find_last_of('.'));	

		m_pMyNet->m_model = m_pMyNet->loadModel(modelName);
		if (!m_pMyNet->m_model) {
			AfxMessageBox(_T("Erreur lors du chargement du modèle."));
			m_BtLancer.EnableWindow(true);
			m_BtOk.EnableWindow(true);
			m_Texte_Train = L"Opération annulée";
			UpdateData(false);
			pDialog->DestroyWindow();
			return;
		}
	}


	m_pMyNet->trainModelDyn(m_Iter, m_LearningRate, m_Batch, m_Periode, modelFilePath.string());
	m_pMyNet->saveModel(m_pMyNet->m_model, modelFilePath.string(), namesFilePath.string());
	m_BtOk.EnableWindow(true);
	m_BtLancer.EnableWindow(true);

	m_Texte_Train.Format(L"Apprentissage terminé\nAccuracy: %.2f%%\nLoss: %.4f", m_pMyNet->m_Accuracy * 100, m_pMyNet->m_Loss);
	//m_Texte_Train = L"Apprentissage terminé\nModèle sauvegardé";
	UpdateData(false);

	//MessageBox(L"Apprentissage termine", L"Succes", MB_OK);
	//fermer la boîte de dialogue
	pDialog->DestroyWindow();
	// Libérer la mémoire
	//delete pDialog;

}

void CDialTrain::OnBnClickedButtonLancerFromModel()
{
	m_FromScratch = false;
	OnBnClickedButtonLancer();
	m_FromScratch = true;
}

void CDialTrain::OnBnClickedButtonGenerepython()
{
	// TODO: ajoutez ici le code de votre gestionnaire de commande
	UpdateData(TRUE);
	m_Texte_Train = L"Python generation\n using Mistral AI";
	UpdateData(false);

	// Désactiver les boutons
	m_BtLancer.EnableWindow(FALSE);
	m_BtOk.EnableWindow(FALSE);

	CFileDialog fileDialog(TRUE, _T("configCNN"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Config Files (*.configCNN)|*.configCNN||"));
	fileDialog.m_ofn.lpstrTitle = _T("Charger la configuration");

	if (fileDialog.DoModal() != IDOK)
	{
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);

		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}
	CString configFilePath = fileDialog.GetPathName();
	std::string stdconfigFilePath = CT2A(configFilePath.GetString());
	std::filesystem::path configPath(configFilePath.GetString());
	std::filesystem::path configDir = configPath.parent_path();

	// Charger la configuration à partir du fichier
	std::vector<ConvLayerParams> layers;
	std::vector<DenseLayerParams> Denselayers;
	if (!m_pMyNet->loadConfig(stdconfigFilePath, layers, Denselayers))
	{
		AfxMessageBox(_T("Erreur lors du chargement de la configuration."));
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}
	if (layers.size() == 0)
	{
		AfxMessageBox(_T("Erreur: aucune couche de convolution trouvée dans le fichier de configuration."));
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}
	// Afficher la boîte de dialogue CCNNDialog sous forme de boîte non modale
	CCNNDialog* pDialog = new CCNNDialog(m_pMyNet->m_input_height, m_pMyNet->m_input_width);
	pDialog->Create(IDD_CNN_DIALOG, this);
	pDialog->ShowWindow(SW_SHOW);

	// Reconstruire la liste à partir du fichier de configuration
	pDialog->SetLayers(layers, Denselayers);
	pDialog->m_Visu.SetLayers(layers, Denselayers);
	pDialog->UpdateTotalParams();

	//désactiver les boutons de la boîte de dialogue pDialog
	pDialog->m_BtConv.EnableWindow(FALSE);
	pDialog->m_BtDense.EnableWindow(FALSE);
	pDialog->m_BtSave.EnableWindow(FALSE);

	MessageBoxA(NULL, "Selectionnez le dossier qui contient les sous-dossiers train et test", "Selectionner le dossier", MB_OK);

	CFolderPickerDialog folderPickerDialog(NULL, OFN_FILEMUSTEXIST, this, 0);
	if (folderPickerDialog.DoModal() != IDOK) {
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Opération annulée";
		UpdateData(false);
		return;
	}

	CString folderPath = folderPickerDialog.GetPathName();
	std::filesystem::path rootPath(folderPath.GetString());

	int NbClasses;

	bool read = m_pMyNet->loadDataset(rootPath.string()); // fixe le nombre de classes et les tensors
	if (!read)
	{
		AfxMessageBox(_T("Directory structure should be	\ntrain\n  classe1\n   image1.jpg\n   image2.jpg\n  classe2\n   image1.jpg\n...\n\ntest\n  classe1\n   image1.jpg\n   image2.jpg\netc."));
		m_BtLancer.EnableWindow(true);
		m_BtOk.EnableWindow(true);
		m_Texte_Train = L"Bad Directory structure";
		UpdateData(false);
		pDialog->DestroyWindow();

		return;
	}

	std::filesystem::path modelFilePath = configDir / std::string(CT2A(m_NomModel));
	std::filesystem::path namesFilePath = configDir / (modelFilePath.stem().string() + ".names");

	if (m_Batch > m_pMyNet->m_train_image_paths.size()) m_Batch = m_pMyNet->m_train_image_paths.size();
	m_pMyNet->m_model = m_pMyNet->createModel(layers, Denselayers, m_pMyNet->m_input_height, m_pMyNet->m_input_width, m_pMyNet->m_nb_classes);

	CMistral Mistral;
	size_t lastDot = stdconfigFilePath.find_last_of('.');
	std::string filePathWithoutExt = stdconfigFilePath.substr(0, lastDot);
	string prompt2 = Mistral.generateTrainingPrompt(layers, Denselayers, m_pMyNet->m_input_height, m_pMyNet->m_input_width, m_pMyNet->m_nb_classes, rootPath.string(), m_Batch, m_Iter, m_LearningRate, true, modelFilePath.string());
	Mistral.sendRequestToMistralAPI(prompt2, filePathWithoutExt + "_training.py");
	// afficher le message de succès
	m_Texte_Train.Format(L"Python file generated\n");
	UpdateData(false);
	m_BtOk.EnableWindow(true);
	m_BtLancer.EnableWindow(true);
	pDialog->DestroyWindow();

}
