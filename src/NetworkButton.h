
#pragma once
#include <afxwin.h>
#include <vector>
#include "CNetcv.h"

class CNetworkButton : public CButton
{
public:
    CNetworkButton();
    virtual ~CNetworkButton();

    void SetLayers(const std::vector<ConvLayerParams>& convLayers, const std::vector<DenseLayerParams>& denseLayers);

protected:
    afx_msg void OnPaint();
    afx_msg void OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
    afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);
    DECLARE_MESSAGE_MAP()

private:
    std::vector<ConvLayerParams> m_convLayers;
    std::vector<DenseLayerParams> m_denseLayers;
    int m_scrollPos;
    int m_totalHeight;
    void DrawLayer(CDC* pDC, const CRect& rect, const CString& text, COLORREF color);
    void DrawActivation(CDC* pDC, const CRect& rect, const CString& text);
    void UpdateScrollBar();
};
