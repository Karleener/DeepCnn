#include "pch.h"
#include "NetworkButton.h"

CNetworkButton::CNetworkButton() : m_scrollPos(0), m_totalHeight(0)
{
}

CNetworkButton::~CNetworkButton()
{
}

BEGIN_MESSAGE_MAP(CNetworkButton, CButton)
    ON_WM_PAINT()
    ON_WM_VSCROLL()
    ON_WM_MOUSEWHEEL()
END_MESSAGE_MAP()

void CNetworkButton::SetLayers(const std::vector<ConvLayerParams>& convLayers, const std::vector<DenseLayerParams>& denseLayers)
{
    m_convLayers = convLayers;
    m_denseLayers = denseLayers;
    Invalidate(); // Redessiner le bouton
    UpdateScrollBar();
}
void CNetworkButton::OnPaint()
{
    int espace = 5;
    CPaintDC dc(this); // contexte de périphérique pour la peinture

    CRect rect;
    GetClientRect(&rect);
    // Redessiner le fond du bouton
    dc.FillSolidRect(&rect, GetSysColor(COLOR_3DFACE));
    // Créer une police plus petite
    CFont font;
    font.CreatePointFont(80, _T("Arial")); // Taille de 8 points
    CFont* pOldFont = dc.SelectObject(&font);

    // Obtenir la hauteur de la police
    TEXTMETRIC tm;
    dc.GetTextMetrics(&tm);
    int textHeight = tm.tmHeight;

    // Définir la hauteur des rectangles légèrement plus grande que la hauteur du texte
    int layerHeight = textHeight + 15;
    int layerWidth = rect.Width() ;
	int Hauteurmax = rect.Height() - layerHeight / 2 + espace;;

    int y = -m_scrollPos;


    for (const auto& layer : m_convLayers)
    {
        if (y + layerHeight > Hauteurmax)      break;
        CRect layerRect(rect.left + layerWidth, y, rect.right - layerWidth, y + layerHeight);
        CString text;
        text.Format(_T("Conv: In=%d, Out=%d, K=%d, S=%d, P=%d"), layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding);
        DrawLayer(&dc, layerRect, text, RGB(173, 216, 230)); // Bleu clair

        y += layerHeight + espace; // Ajouter un espace entre les couches
        if (y + layerHeight > Hauteurmax)      break;
        CRect activationRect(rect.left + layerWidth, y, rect.right - layerWidth, y + layerHeight / 2);
        DrawActivation(&dc, activationRect, _T("ReLU"));

        y += layerHeight / 2 + espace; // Ajouter un espace entre les couches
    }
    // Ajouter la couche flatten
  // Ajouter la couche flatten
    if (y + layerHeight <= Hauteurmax)
    {
        CRect flattenRect(rect.left + layerWidth, y, rect.right - layerWidth, y + layerHeight);
        DrawLayer(&dc, flattenRect, _T("Flatten"), RGB(255, 255, 255)); // Blanc

        y += layerHeight + espace; // Ajouter un espace entre les couches
    }

    for (const auto& layer : m_denseLayers)
    {
        if (y + layerHeight > Hauteurmax)      break;
        CRect layerRect(rect.left + layerWidth, y, rect.right - layerWidth, y + layerHeight);
        CString text;
        text.Format(_T("Dense: Neurons=%d"), layer.nb_neurons);
        DrawLayer(&dc, layerRect, text, RGB(255, 182, 193)); // Rose clair

        y += layerHeight + espace; // Ajouter un espace entre les couches
        if (y + layerHeight > Hauteurmax)      break;
        CRect activationRect(rect.left + layerWidth, y, rect.right - layerWidth, y + layerHeight / 2);
        DrawActivation(&dc, activationRect, layer.activation_type == 0 ? _T("ReLU") : _T("Sigmoid"));

        y += layerHeight / 2 + espace; // Ajouter un espace entre les couches
    }
    // Ajouter la couche softmax
    if (y + layerHeight <= Hauteurmax)
    {
        CRect softmaxRect(rect.left + layerWidth, y, rect.right - layerWidth, y + layerHeight);
        DrawLayer(&dc, softmaxRect, _T("Softmax"), RGB(255, 255, 0)); // Jaune

        y += layerHeight; // Ajouter un espace entre les couches
    }
    // Calculer la hauteur totale du contenu
    m_totalHeight = y + layerHeight;

    // Restaurer l'ancienne police
    dc.SelectObject(pOldFont);
}


void CNetworkButton::DrawLayer(CDC* pDC, const CRect& rect, const CString& text, COLORREF color)
{
    CBrush brush(color);
    pDC->FillRect(rect, &brush);
    // Dessiner le bord noir
    // Créer une police plus petite
    CFont font;
    font.CreatePointFont(80, _T("Arial")); // Taille de 8 points
    CFont* pOldFont = pDC->SelectObject(&font);

    pDC->DrawText(text, const_cast<CRect*>(&rect), DT_CENTER | DT_VCENTER | DT_SINGLELINE);
   // pDC->FrameRect(&rect, &CBrush(RGB(0, 0, 0)));

    pDC->SelectObject(pOldFont);

}

void CNetworkButton::DrawActivation(CDC* pDC, const CRect& rect, const CString& text)
{
    CBrush brush(RGB(144, 238, 144)); // Vert clair
    pDC->FillRect(rect, &brush);
    // Dessiner le bord noir
  //  pDC->FrameRect(&rect, &CBrush(RGB(0, 0, 0)));
    // Créer une police plus petite
    CFont font;
    font.CreatePointFont(80, _T("Arial")); // Taille de 8 points
    CFont* pOldFont = pDC->SelectObject(&font);
    pDC->DrawText(text, const_cast<CRect*>(&rect), DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    pDC->SelectObject(pOldFont);
}


void CNetworkButton::OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
    int nDelta;
    int nMaxPos = m_totalHeight - 1;

    switch (nSBCode)
    {
    case SB_LINEUP:
        nDelta = -10;
        break;
    case SB_LINEDOWN:
        nDelta = 10;
        break;
    case SB_PAGEUP:
        nDelta = -50;
        break;
    case SB_PAGEDOWN:
        nDelta = 50;
        break;
    case SB_THUMBTRACK:
        nDelta = (int)nPos - m_scrollPos;
        break;
    default:
        return;
    }

    int nNewPos = m_scrollPos + nDelta;
    if (nNewPos < 0)
        nNewPos = 0;
    else if (nNewPos > nMaxPos)
        nNewPos = nMaxPos;

    if (nNewPos != m_scrollPos)
    {
        m_scrollPos = nNewPos;
        SetScrollPos(SB_VERT, m_scrollPos);
        Invalidate();
    }
}

BOOL CNetworkButton::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
    int nDelta = -zDelta / 120 * 50;
    int nNewPos = m_scrollPos + nDelta;
    int nMaxPos = m_totalHeight - 1;

    if (nNewPos < 0)
        nNewPos = 0;
    else if (nNewPos > nMaxPos)
        nNewPos = nMaxPos;

    if (nNewPos != m_scrollPos)
    {
        m_scrollPos = nNewPos;
        SetScrollPos(SB_VERT, m_scrollPos);
        Invalidate();
    }

    return TRUE;
}

void CNetworkButton::UpdateScrollBar()
{
    SCROLLINFO si;
    si.cbSize = sizeof(SCROLLINFO);
    si.fMask = SIF_RANGE | SIF_PAGE | SIF_POS;
    si.nMin = 0;
    si.nMax = m_totalHeight - 1;
    si.nPage = 100;
    si.nPos = m_scrollPos;
    SetScrollInfo(SB_VERT, &si, TRUE);
}
