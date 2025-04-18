// Courbe.cpp : implementation file
//

#include "pch.h"
#include "Courbe.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CCourbe

IMPLEMENT_DYNCREATE(CCourbe, CFrameWnd)

CCourbe::CCourbe()
{
	minmaxdefined = false; imin = 0.0; imax = 0.0;
}

CCourbe::~CCourbe()
{
}


BEGIN_MESSAGE_MAP(CCourbe, CFrameWnd)
	//{{AFX_MSG_MAP(CCourbe)
	ON_WM_PAINT()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCourbe message handlers

void CCourbe::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	Dessine(m_x,m_y,m_z,m_x.size());

	// Do not call CFrameWnd::OnPaint() for painting messages
}



int CCourbe::Dessine(const vector<double>& x, const vector<double>& y,  const vector<double>& z, int limit=0)
{
	//Invalidate(true);
	if (x.size()<2 || y.size()<2 || z.size() < 2) return 0;
	CClientDC dc(this); // device context for painting

	CPen StyloNoir(PS_SOLID, 2, RGB(0, 0, 0));
	CPen StyloRouge(PS_SOLID, 1, RGB(255, 0, 0));
	CPen StyloBleu(PS_SOLID, 2, RGB(0, 0, 255));
	CRect MonRect;
	GetClientRect(&MonRect);
	dc.SelectObject(&StyloNoir);

	CBrush brush;
	brush.CreateSolidBrush(RGB(210, 210, 210));
	dc.FillRect(MonRect, &brush);

	int PourcentMax = 50;
	int Hauteur = MonRect.Height();
	int Largeur = MonRect.Width();
	int i;
	int Taille = x.size();
	double maxy = y[0];
	double miny = y[0];
	double xf, yf, hf;
	//m_x.clear();
	//m_y.clear();
	//m_x.resize(Taille);
	//m_y.resize(Taille);
	m_x = x;
	m_y = y;
	m_z = z;

	for (i = 0; i < Taille; i++)
	{
		if (y[i] > maxy) maxy = y[i];
		if (y[i] < miny) miny = y[i];
		//m_x[i] = x[i];
		//m_y[i] = y[i];
	}

	dc.SelectObject(&StyloBleu);

	yf = Hauteur * ((y[0] - miny) / (maxy - miny));
	hf = Hauteur - yf;
	dc.MoveTo(0, hf);
	i = 1;

	if (limit != 0) limit = std::min(limit, (int)x.size());
	//else limit = Taille;
	do
	{
		yf = Hauteur * ((y[i] - miny) / (maxy - miny));
		hf = Hauteur - yf;
		xf = ((i+1) * Largeur) / (Taille);
		dc.LineTo(xf, hf);
	}
	while (++i <= limit);

	yf = Hauteur * (z[0]);
	hf = Hauteur - yf;
	dc.SelectObject(&StyloRouge);
	dc.MoveTo(0, hf);
	i = 1;
	do
	{
		yf = Hauteur * z[i]; // accuracy entre 0 et 1
		hf = Hauteur - yf;
		xf = ((i + 1) * Largeur) / (Taille );
		dc.LineTo(xf, hf);
	} while (++i <= limit);

	if (minmaxdefined)
	{
		dc.SelectObject(&StyloRouge);
		xf = (imin * Largeur) / (Taille );
		dc.MoveTo(xf, 0);
		dc.LineTo(xf, Hauteur);
		xf = (imax * Largeur) / (Taille );
		dc.MoveTo(xf, 0);
		dc.LineTo(xf, Hauteur);
	}
	ReleaseDC(&dc);
	return 0;
}