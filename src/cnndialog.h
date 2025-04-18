// CNNDialog.h
#pragma once
#include "resource.h"
#include "afxdialogex.h"
#include "CNetcv.h"
#include "NetworkButton.h"
#include <vector>


class CCNNDialog : public CDialogEx
{
    DECLARE_DYNAMIC(CCNNDialog)
public:
    CCNNDialog(int inputHeight, int inputWidth, CWnd* pParent = nullptr);

#ifdef AFX_DESIGN_TIME
    enum { IDD = IDD_CNN_DIALOG };
#endif

protected:
    virtual void DoDataExchange(CDataExchange* pDX);

protected:
    DECLARE_MESSAGE_MAP()

private:
    std::vector<ConvLayerParams> m_layers;
    std::vector<DenseLayerParams> m_denseLayers;
    CNetcv m_netcv;
    int m_inputHeight;
    int m_inputWidth;
    int m_totalParams;
    bool m_isModified;

public:
   // void UpdateLayerList();
    void UpdateDenseLayerList();
    afx_msg void OnBnClickedAddLayer();
    afx_msg void OnBnClickedButtonSaveConfig();
    afx_msg void OnBnClickedButtonOk();
    void SetLayers(const std::vector<ConvLayerParams>& layers, std::vector<DenseLayerParams> &Denselayers);
    void UpdateTotalParams();
    bool ValidateLayer(ConvLayerParams& layer, int& currentHeight, int& currentWidth);

    virtual BOOL OnInitDialog();
    int m_TypeActivation;
    int m_NbNeurons;
    afx_msg void OnBnClickedButtonAddMlpLayer();
    CNetworkButton m_Visu;
    CButton m_BtSave;
    CButton m_BtDense;
    CButton m_BtConv;
};