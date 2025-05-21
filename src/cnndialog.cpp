
// CNNDialog.cpp
#include "pch.h"
#include "CNNDialog.h"
#include "afxdialogex.h"
#include "resource.h"

IMPLEMENT_DYNAMIC(CCNNDialog, CDialogEx)

CCNNDialog::CCNNDialog(int inputHeight, int inputWidth, CWnd* pParent /*=nullptr*/)
    : CDialogEx(IDD_CNN_DIALOG, pParent), m_inputHeight(inputHeight), m_inputWidth(inputWidth), m_totalParams(0)
    , m_TypeActivation(0)
    , m_NbNeurons(10)
	, m_isModified(false)
{

}

void CCNNDialog::DoDataExchange(CDataExchange* pDX)
{
    CDialogEx::DoDataExchange(pDX);
    DDX_Radio(pDX, IDC_RADIO1, m_TypeActivation);
    DDX_Text(pDX, IDC_EDIT1, m_NbNeurons);
    DDV_MinMaxInt(pDX, m_NbNeurons, 1, 10000);
    DDX_Control(pDX, IDC_BUTTON_VISU, m_Visu);
    DDX_Control(pDX, IDC_BUTTON_SAVE_CONFIG, m_BtSave);
    DDX_Control(pDX, IDC_BUTTON_ADD_MLP_LAYER, m_BtDense);
    DDX_Control(pDX, IDC_BUTTON_ADD_LAYER, m_BtConv);
}

BEGIN_MESSAGE_MAP(CCNNDialog, CDialogEx)
    ON_BN_CLICKED(IDC_BUTTON_ADD_LAYER, &CCNNDialog::OnBnClickedAddLayer)
    ON_BN_CLICKED(IDC_BUTTON_SAVE_CONFIG, &CCNNDialog::OnBnClickedButtonSaveConfig)
    ON_BN_CLICKED(IDC_BUTTON_OK, &CCNNDialog::OnBnClickedButtonOk)
    ON_BN_CLICKED(IDC_BUTTON_ADD_MLP_LAYER, &CCNNDialog::OnBnClickedButtonAddMlpLayer)
    ON_BN_CLICKED(IDC_BUTTON_PYTHON, &CCNNDialog::OnBnClickedButtonPython)
END_MESSAGE_MAP()

void CCNNDialog::OnBnClickedAddLayer()
{
    ConvLayerParams layer;
    CString strValue;

    GetDlgItemText(IDC_EDIT_IN_CHANNELS, strValue);
    layer.in_channels = _ttoi(strValue);

    GetDlgItemText(IDC_EDIT_OUT_CHANNELS, strValue);
    layer.out_channels = _ttoi(strValue);

    GetDlgItemText(IDC_EDIT_KERNEL_SIZE, strValue);
    layer.kernel_size = _ttoi(strValue);

    GetDlgItemText(IDC_EDIT_STRIDE, strValue);
    layer.stride = _ttoi(strValue);

    GetDlgItemText(IDC_EDIT_PADDING, strValue);
    layer.padding = _ttoi(strValue);

    layer.use_pool = (IsDlgButtonChecked(IDC_CHECK_USE_POOL) == BST_CHECKED);

    GetDlgItemText(IDC_EDIT_POOL_SIZE, strValue);
    layer.pool_size = _ttoi(strValue);

   
    strValue.Format(_T("%d"), layer.out_channels);
    SetDlgItemText(IDC_EDIT_IN_CHANNELS, strValue);



    // Valider la couche avant de l'ajouter
    int currentHeight = m_inputHeight;
    int currentWidth = m_inputWidth;
    for (const auto& existingLayer : m_layers)
    {
        ValidateLayer(const_cast<ConvLayerParams&>(existingLayer), currentHeight, currentWidth);
    }

    if (ValidateLayer(layer, currentHeight, currentWidth))
    {
        m_layers.push_back(layer);
      //  UpdateLayerList();
        UpdateTotalParams();
        strValue.Format(_T("%d"), layer.out_channels);
        SetDlgItemText(IDC_EDIT_IN_CHANNELS, strValue);

        m_Visu.SetLayers(m_layers, m_denseLayers);
    }
    else
    {
        AfxMessageBox(_T("Erreur : La couche ajoutée n'est pas valide et ne peut pas être ajustée automatiquement."));
    }
	m_isModified = true;

}


void CCNNDialog::OnBnClickedButtonSaveConfig()
{
    CFileDialog fileDialog(FALSE, _T("configCNN"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Config Files (*.configCNN)|*.configCNN||"));
    fileDialog.m_ofn.lpstrTitle = _T("Enregistrer la configuration");
	m_netcv.m_input_height = m_inputHeight;
	m_netcv.m_input_width = m_inputWidth;
	if (m_layers.size() == 0)
	{
		AfxMessageBox(_T("Erreur : aucune couche de convolution trouvée."));
		return;
	}
    if (fileDialog.DoModal() == IDOK)
    {
        CString filePath = fileDialog.GetPathName();
        std::string stdFilePath = CT2A(filePath.GetString());
        if (m_netcv.saveConfig(m_layers, m_denseLayers, stdFilePath,false ))
        {
            AfxMessageBox(_T("Configuration sauvegardee."));
            m_isModified = false;
        }
        else
        {
            AfxMessageBox(_T("Erreur lors de la sauvegarde de la configuration."));
        }
    }
   
}

void CCNNDialog::OnBnClickedButtonOk()
{
    // TODO: ajoutez ici le code de votre gestionnaire de notification de contrôle

    if (m_isModified)
    {
        int test = MessageBox(L"La sauvegarde n'est pas automatique \n Voulez-vous vraiment quitter?", L"Quitter", MB_YESNO | MB_ICONQUESTION);
        if (test == IDYES) CDialogEx::OnOK(); return;
    }
	CDialogEx::OnOK();
   
}

void CCNNDialog::SetLayers(const std::vector<ConvLayerParams>& layers, std::vector<DenseLayerParams> &Denselayers)
{
    m_layers = layers;
	m_denseLayers = Denselayers;
}

void CCNNDialog::UpdateTotalParams()
{
    m_totalParams = 0;
    int currentHeight = m_inputHeight;
    int currentWidth = m_inputWidth;

    for (const auto& layer : m_layers)
    {
        int convParams = layer.out_channels * layer.in_channels * layer.kernel_size * layer.kernel_size;
        int biasParams = layer.out_channels;
        m_totalParams += convParams + biasParams;

        // Calculer les dimensions de sortie
        currentHeight = (currentHeight + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        currentWidth = (currentWidth + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;

        if (layer.use_pool)
        {
            currentHeight /= layer.pool_size;
            currentWidth /= layer.pool_size;
        }
    }

    // Comptabiliser les paramètres des couches denses
    for (const auto& layer : m_denseLayers)
    {
        int denseParams = layer.nb_neurons * (currentHeight * currentWidth) + layer.nb_neurons; // Poids + biais
        m_totalParams += denseParams;

        // Après la première couche dense, les dimensions deviennent 1D
        currentHeight = 1;
        currentWidth = 1;
    }

    CString strTotalParams;
    strTotalParams.Format(_T("Parametres: %d"), m_totalParams);
    SetDlgItemText(IDC_STATIC_TOTAL_PARAMS, strTotalParams);
}
BOOL CCNNDialog::OnInitDialog()
{
    CDialogEx::OnInitDialog();

    // TODO:  Ajoutez ici une initialisation supplémentaire
    SetDlgItemText(IDC_EDIT_IN_CHANNELS, _T("3"));
    SetDlgItemText(IDC_EDIT_OUT_CHANNELS, _T("32"));
    SetDlgItemText(IDC_EDIT_KERNEL_SIZE, _T("3"));
    SetDlgItemText(IDC_EDIT_STRIDE, _T("1"));
    SetDlgItemText(IDC_EDIT_PADDING, _T("1"));
    CheckDlgButton(IDC_CHECK_USE_POOL, BST_CHECKED);
    SetDlgItemText(IDC_EDIT_POOL_SIZE, _T("2"));

    return TRUE;  // return TRUE unless you set the focus to a control
    // EXCEPTION : les pages de propriétés OCX devraient retourner FALSE
}


bool CCNNDialog::ValidateLayer(ConvLayerParams& layer, int& currentHeight, int& currentWidth)
{
    // Calculer les dimensions de sortie de la convolution
    int newHeight = (currentHeight + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
    int newWidth = (currentWidth + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;

    // Vérifier si les dimensions de sortie sont valides
    if (newHeight <= 0 || newWidth <= 0)
    {
        // Proposer un ajustement automatique
        layer.kernel_size = std::min(currentHeight, currentWidth);
        newHeight = (currentHeight + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        newWidth = (currentWidth + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
    }

    // Vérifier si les dimensions de sortie après pooling sont valides
    if (layer.use_pool)
    {
        newHeight /= layer.pool_size;
        newWidth /= layer.pool_size;

        if (newHeight <= 0 || newWidth <= 0)
        {
            // Proposer un ajustement automatique
            layer.pool_size = std::min(currentHeight, currentWidth);
            newHeight = (currentHeight + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
            newWidth = (currentWidth + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
            newHeight /= layer.pool_size;
            newWidth /= layer.pool_size;
        }
    }

    // Mettre à jour les dimensions actuelles
    currentHeight = newHeight;
    currentWidth = newWidth;

    return true;
}
void CCNNDialog::OnBnClickedButtonAddMlpLayer()
{
    UpdateData(TRUE);
    DenseLayerParams layer;
    layer.nb_neurons = m_NbNeurons;
    layer.activation_type = m_TypeActivation;

    // Ajouter la couche dense à la liste
    m_denseLayers.push_back(layer);

    // Mettre à jour l'affichage de la liste des couches
    UpdateTotalParams();
    m_Visu.SetLayers(m_layers, m_denseLayers);
    m_isModified = true;

}
void CCNNDialog::UpdateDenseLayerList()
{
	CListBox* pListBox = (CListBox*)GetDlgItem(IDC_LIST_STRUCTURE);
	pListBox->ResetContent();
	for (const auto& layer : m_denseLayers)
	{
		CString strLayer;
		strLayer.Format(_T("Neurons: %d, Activation: %d"), layer.nb_neurons, layer.activation_type);
		pListBox->AddString(strLayer);
	}
}
void CCNNDialog::OnBnClickedButtonPython()
{
    CFileDialog fileDialog(FALSE, _T("configCNN"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Config Files (*.configCNN)|*.configCNN||"));
    fileDialog.m_ofn.lpstrTitle = _T("Enregistrer la configuration");
    m_netcv.m_input_height = m_inputHeight;
    m_netcv.m_input_width = m_inputWidth;
    if (m_layers.size() == 0)
    {
        AfxMessageBox(_T("Erreur : aucune couche de convolution trouvée."));
        return;
    }
    if (fileDialog.DoModal() == IDOK)
    {
        CString filePath = fileDialog.GetPathName();
        std::string stdFilePath = CT2A(filePath.GetString());
        if (m_netcv.saveConfig(m_layers, m_denseLayers, stdFilePath, true))
        {
            AfxMessageBox(_T("Configuration sauvegardee."));
            m_isModified = false;
        }
        else
        {
            AfxMessageBox(_T("Erreur lors de la sauvegarde de la configuration."));
        }
    }
}
