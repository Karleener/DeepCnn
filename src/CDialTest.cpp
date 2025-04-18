// CDialTest.cpp : fichier d'implémentation
//

#include "pch.h"
#include "DeepCnn2025.h"
#include "afxdialogex.h"
#include "CDialTest.h"
#include "CNetcv.h"
#include  <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
// boîte de dialogue de CDialTest

IMPLEMENT_DYNAMIC(CDialTest, CDialogEx)

CDialTest::CDialTest(CWnd* pParent /*=nullptr*/, CNetcv *pMyNet )
	: m_pMyNet(pMyNet),CDialogEx(IDD_DIALOG_TEST, pParent)
	, m_Texte(_T(""))
	, m_blob(28)
	, m_Tol(80)
	, m_boolEdges(TRUE)
{

}


CDialTest::~CDialTest()
{
}

void CDialTest::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_STATIC_TEXTE, m_Texte);
	DDX_Text(pDX, IDC_EDIT_BLOB2, m_blob);
	DDV_MinMaxInt(pDX, m_blob, 4, 512);
	DDX_Text(pDX, IDC_EDIT_TOLER, m_Tol);
	DDV_MinMaxInt(pDX, m_Tol, 30, 100);
	DDX_Check(pDX, IDC_CHECK1, m_boolEdges);
}


BEGIN_MESSAGE_MAP(CDialTest, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON_CHOIXMODEL, &CDialTest::OnBnClickedButtonChoixmodel)
	ON_BN_CLICKED(IDC_BUTTON_CHOIXIMAGE, &CDialTest::OnBnClickedButtonChoiximage)
	ON_BN_CLICKED(IDC_BUTTON_CHOIXIMAGE_SEG, &CDialTest::OnBnClickedButtonChoiximageSeg)
	ON_BN_CLICKED(IDC_BUTTON_CHOIXIMAGE_GT, &CDialTest::OnBnClickedButtonChoiximageGt)
	ON_BN_CLICKED(IDC_BUTTON_CHOIXIMAGE_BRAILLE, &CDialTest::OnBnClickedButtonCreateTrainTest)
END_MESSAGE_MAP()


// gestionnaires de messages de CDialTest

void CDialTest::OnBnClickedButtonChoixmodel()
{
	// lecture du modele avec choix du fichier
	CFileDialog fileDialog(TRUE, _T("model_cnn.pt"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Model Files (*.pt)|*.pt||"));
	fileDialog.m_ofn.lpstrTitle = _T("Charger le modèle");
	if (fileDialog.DoModal() != IDOK) return;
	filePath = CT2A(fileDialog.GetPathName());
	// si m_Mynet.m_model n'est pas vide libérer la mémoire
	if (m_pMyNet->m_model) {
		m_pMyNet->m_model.reset();
	}
	std::string modelName = filePath.substr(0, filePath.find_last_of('.'));
	m_pMyNet->loadModel(modelName); // extention ajoutee dans la fonction
	m_Texte = L"modele : " + fileDialog.GetFileName();
	UpdateData(false);

}

void CDialTest::OnBnClickedButtonChoiximage()
{
	m_Texte = "";
	UpdateData(false);
	//if (filePath=="") {
	//	AfxMessageBox(_T("Veuillez d'abord sélectionner un modèle!"), MB_ICONERROR);
	//	OnBnClickedButtonChoixmodel();
	//}
	//if (m_MyNet.m_model) {
	//	m_MyNet.m_model.reset();
	//}
	//m_MyNet.loadModel(filePath);
	if (!m_pMyNet->m_model) OnBnClickedButtonChoixmodel();
	// Afficher la boîte de dialogue pour sélectionner une image
	CFileDialog dlgFile(TRUE, NULL, NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
		_T("Image Files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|All Files (*.*)|*.*||"), NULL);

	if (dlgFile.DoModal() != IDOK) {
		return; // L'utilisateur a annulé
	}

	// Récupérer le chemin de l'image sélectionnée
	CString filePath = dlgFile.GetPathName();
	std::string imagePath = CT2A(filePath);

	// Charger l'image avec OpenCV (supposant que vous utilisez OpenCV)
	cv::Mat image = cv::imread(imagePath);
	if (image.empty()) {
		AfxMessageBox(_T("Erreur lors du chargement de l'image!"), MB_ICONERROR);
		return;
	}

	// Prétraitement de l'image (redimensionnement, normalisation, etc.)
	cv::Mat resized_image;
	cv::resize(image, resized_image, cv::Size(m_pMyNet->m_input_height, m_pMyNet->m_input_width)); // Ajustez selon votre modèle

	// Convertir de BGR à RGB si nécessaire
//	cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

	// Normaliser les valeurs de pixel (0-255 -> 0-1)
	//resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
	std::map<std::string, float> results;
	results = m_pMyNet->testImageInference(m_pMyNet->m_model, resized_image);
	// trouver la classe avec la probabilité maximale
	
	std::string predicted_class;
	float max_prob = 0.0f;
	for (const auto& pair : results) {
		if (pair.second > max_prob) {
			max_prob = pair.second;
			predicted_class = pair.first;
		}
	}
 
	// Afficher le résultat
	std::ostringstream result_stream;
	result_stream << "Classe predite: " << predicted_class << std::endl;

	// Convertir le message en CString pour AfxMessageBox
	CString result_message(result_stream.str().c_str());
	//AfxMessageBox(result_message, MB_ICONINFORMATION);
	m_Texte = result_message;
	UpdateData(false);
	// Optionnel: Afficher l'image avec la prédiction
	cv::Mat display_image = cv::imread(imagePath);

	double aspect_ratio = static_cast<double>(display_image.rows) / display_image.cols;
	int new_width = 768;
	int new_height = static_cast<int>(new_width * aspect_ratio);
	cv::resize(display_image, display_image, cv::Size(new_width, new_height));


	cv::putText(display_image, predicted_class, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
	cv::imshow("Prediction", display_image);
	cv::waitKey(0);
}

//void CDialTest::OnBnClickedButtonChoiximageSeg()
//{
//	// Vérifier si un modèle est chargé, sinon demander à l'utilisateur d'en charger un
//	if (!m_pMyNet->m_model) {
//		OnBnClickedButtonChoixmodel();
//		if (!m_pMyNet->m_model) {
//			AfxMessageBox(_T("Aucun modèle chargé. Opération annulée."), MB_ICONERROR);
//			return;
//		}
//	}
//
//	// Afficher la boîte de dialogue pour sélectionner une image
//	CFileDialog dlgFile(TRUE, NULL, NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
//		_T("Image Files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|All Files (*.*)|*.*||"), NULL);
//
//	if (dlgFile.DoModal() != IDOK) {
//		return; // L'utilisateur a annulé
//	}
//
//	// Récupérer le chemin de l'image sélectionnée
//	CString filePath = dlgFile.GetPathName();
//	std::string imagePath = CT2A(filePath);
//
//	// Charger l'image avec OpenCV
//	cv::Mat image = cv::imread(imagePath);
//	if (image.empty()) {
//		AfxMessageBox(_T("Erreur lors du chargement de l'image!"), MB_ICONERROR);
//		return;
//	}
//
//	// Paramètres de segmentation
//	int window_size = 32; // Taille initiale des sous-fenêtres (32x32)
//	int stride = 8;      // Stride initial (16 pixels)
//	int num_classes = m_pMyNet->m_nb_classes;
//
//	// Créer une image de sortie pour la segmentation
//	cv::Mat segmented_image(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
//
//	// Parcourir l'image avec des sous-fenêtres glissantes
//	for (int y = 0; y <= image.rows - window_size; y += stride) {
//		for (int x = 0; x <= image.cols - window_size; x += stride) {
//			// Extraire la sous-fenêtre
//			cv::Rect window(x, y, window_size, window_size);
//			cv::Mat sub_window = image(window);
//
//			// Redimensionner la sous-fenêtre à la taille d'entrée du modèle
//			//cv::Mat resized_window;
//			//cv::resize(sub_window, resized_window, cv::Size(m_pMyNet->m_input_width, m_pMyNet->m_input_height));
//
//			// Classer la sous-fenêtre
//			std::map<std::string, float> results = m_pMyNet->testImageInference(m_pMyNet->m_model, sub_window);
//			// Trouver l'indice de la classe avec la probabilité maximale
//			int class_index = -1;
//			float max_prob = 0.0f;
//			for (size_t i = 0; i < m_pMyNet->class_names.size(); ++i) {
//				const std::string& class_name = m_pMyNet->class_names[i];
//				if (results[class_name] > max_prob) {
//					max_prob = results[class_name];
//					class_index = static_cast<int>(i);
//				}
//			}
//
//			// Générer une couleur unique pour chaque classe
//			cv::Vec3b color;
//			color[0] = (class_index * 50) % 256; // Bleu
//			color[1] = (class_index * 100) % 256; // Vert
//			color[2] = (class_index * 150) % 256; // Rouge
//
//			// Colorer les pixels de la sous-fenêtre dans l'image segmentée
//			for (int i = y; i < y + window_size && i < segmented_image.rows; ++i) {
//				for (int j = x; j < x + window_size && j < segmented_image.cols; ++j) {
//					segmented_image.at<cv::Vec3b>(i, j) = color;
//				}
//			}
//		}
//	}
//	// sauvegarder l'image segmentée
//	cv::imwrite("segmented_image.png", segmented_image);
//	// Afficher l'image segmentée
//	cv::imshow("Segmented Image", segmented_image);
//	cv::waitKey(0);
//}

void CDialTest::OnBnClickedButtonChoiximageSeg()
{
	UpdateData(true);
	// Vérifier si un modèle est chargé, sinon demander à l'utilisateur d'en charger un
	if (!m_pMyNet->m_model) {
		OnBnClickedButtonChoixmodel();
		if (!m_pMyNet->m_model) {
			AfxMessageBox(_T("Aucun modèle chargé. Opération annulée."), MB_ICONERROR);
			return;
		}
	}

	// Afficher la boîte de dialogue pour sélectionner une image
	CFileDialog dlgFile(TRUE, NULL, NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
		_T("Image Files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|All Files (*.*)|*.*||"), NULL);

	if (dlgFile.DoModal() != IDOK) {
		return; // L'utilisateur a annulé
	}

	// Récupérer le chemin de l'image sélectionnée
	CString filePath = dlgFile.GetPathName();
	std::string imagePath = CT2A(filePath);

	// Charger l'image avec OpenCV
	cv::Mat image = cv::imread(imagePath);
	if (image.empty()) {
		AfxMessageBox(_T("Erreur lors du chargement de l'image!"), MB_ICONERROR);
		return;
	}

	// Paramètres de segmentation
	int window_size = m_blob; // Taille initiale des sous-fenêtres (32x32)
	int stride = 2;       // Stride initial (8 pixels)
	float iou_threshold = 0.20;//  0.5f; // Seuil pour la suppression des réponses multiples
	int num_classes = m_pMyNet->m_nb_classes;
	uchar Thresh = 100; // seuil pour le contour de l'image
	//float SeuilConf = 0.3f; // seuil de confiance pour la prédiction
	float SeuilConf = static_cast<float>(m_Tol) / 100.0f;
	// contour de l'image pour sélectionner les pixels avec haut gradient
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	cv::Mat edges;
	// canny avec seuil pour trouver beaucoup de contours	

	if (m_boolEdges) cv::Canny(gray_image, edges, 50, 120);
	//cv::imshow("contour", edges);


	// Liste pour stocker les prédictions : {classe, probabilité, coordonnées (x, y, w, h)}
	std::vector<std::tuple<int, float, cv::Rect>> predictions;

	// Parcourir l'image avec des sous-fenêtres glissantes
	for (int y = 0; y <= image.rows - window_size; y += stride) 
	{
		for (int x = 0; x <= image.cols - window_size; x += stride) 
		{
			// Extraire la sous-fenêtre
			// si le contour est inférieur à un seuil on ne traite pas la fenêtre
			if (m_boolEdges)
			{
				if (edges.at<uchar>(y + window_size / 2, x + window_size / 2) < Thresh) continue;
			}

			cv::Rect window(x, y, window_size, window_size);
			cv::Mat sub_window = image(window);

			// Redimensionner la sous-fenêtre à la taille d'entrée du modèle
			cv::Mat resized_window;
			cv::resize(sub_window, resized_window, cv::Size(m_pMyNet->m_input_width, m_pMyNet->m_input_height));

			// Classer la sous-fenêtre
			std::map<std::string, float> results = m_pMyNet->testImageInference(m_pMyNet->m_model, resized_window);

			// Trouver l'indice de la classe avec la probabilité maximale
			int class_index = -1;
			float max_prob = 0.0f;
		
			for (size_t i = 0; i < m_pMyNet->class_names.size(); ++i) 
			{
				const std::string& class_name = m_pMyNet->class_names[i];
				float prob = results[class_name];
				if (prob > max_prob) 
				{
					max_prob = results[class_name];
					class_index = static_cast<int>(i);
				}
			}

			// Ajouter la prédiction à la liste
			if (class_index > 0)  // class 0 is background
			{
				// Vérifier si la probabilité est supérieure à un seuil
				if (max_prob > SeuilConf)
				{ // Ajustez le seuil selon vos besoins
					predictions.emplace_back(class_index, max_prob, window);
				}
			}
		}
	}

	// Suppression des réponses multiples (Non-Maximum Suppression)
	std::vector<std::tuple<int, float, cv::Rect>> final_predictions;
	while (!predictions.empty())
	{
		// Trouver la prédiction avec la probabilité maximale
		auto max_it = std::max_element(predictions.begin(), predictions.end(),
			[](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });
		final_predictions.push_back(*max_it);

		// Supprimer les prédictions qui se chevauchent avec un IoU supérieur au seuil
		cv::Rect max_rect = std::get<2>(*max_it);
		predictions.erase(std::remove_if(predictions.begin(), predictions.end(),
			[&max_rect, iou_threshold](const auto& pred) {
				cv::Rect rect = std::get<2>(pred);
				float intersection = (max_rect & rect).area();
				float union_area = max_rect.area() + rect.area() - intersection;
				float iou = intersection / union_area;
				return iou > iou_threshold;
			}),
			predictions.end());
	}

	// Créer une image de sortie pour la segmentation
	cv::Mat segmented_image(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	// Dessiner les prédictions finales sur l'image segmentée
	for (const auto& [class_index, prob, rect] : final_predictions) {
		// Générer une couleur unique pour chaque classe
		cv::Vec3b color;
		color[0] = ((class_index+1) * 50) % 256; // Bleu
		color[1] = ((class_index+1) * 100) % 256; // Vert
		color[2] = ((class_index+1) * 150) % 256; // Rouge

		// Colorer les pixels de la fenêtre dans l'image segmentée
		for (int i = rect.y; i < rect.y + rect.height && i < segmented_image.rows; ++i) {
			for (int j = rect.x; j < rect.x + rect.width && j < segmented_image.cols; ++j) {
				segmented_image.at<cv::Vec3b>(i, j) = color;
			}
		}
	}

	// Sauvegarder l'image segmentée
	// le nom de l'image semgentee est le nom de l'image d'origine + _semgmentee

	std::string segmented_image_path = imagePath.substr(0, imagePath.find_last_of('.')) + "_segmented.png";
	cv::addWeighted(image, 0.5, segmented_image, 0.5, 0, segmented_image);
	cv::imwrite(segmented_image_path, segmented_image);
	// Afficher l'image segmentée
	cv::imshow("Segmented Image", segmented_image);


	// Filtrer les fenêtres de la classe 1
	//std::vector<std::tuple<int, float, cv::Rect>> class1_windows;
	//for (const auto& [class_index, prob, rect] : final_predictions) {
	//	//if (class_index !=0) { // Classe 1
	//		class1_windows.emplace_back(class_index, prob, rect);
	//	//}
	//}
	// enregistrer la liste des fenetres dans un fichier "fenetres.txt"
	// le nom du fichier est le nom de l'image d'origine + "_fenetres.txt"
	std::string windows_file_path = imagePath.substr(0, imagePath.find_last_of('.')) + "_subwindows.csv";
	std::ofstream file(windows_file_path);
	if (file.is_open()) {
		// Ajouter un en-tête pour le fichier CSV
		file << "class_index,prob,x,y,width,height\n";
		for (const auto& [class_index, prob, rect] : final_predictions) {
			file << class_index << "," << prob << "," << rect.x << "," << rect.y << "," << rect.width << "," << rect.height << "\n";
		}
		file.close();
	}
	else {
		AfxMessageBox(_T("Erreur lors de la sauvegarde des coordonnées des fenêtres!"), MB_ICONERROR);
	}
	m_image_path = imagePath;
	// Appeler la fonction pour incruster les caractères décodés
	//OverlayDecodedBraille(image, class1_windows);
	cv::waitKey(0);

}





void CDialTest::onMouse(int event, int x, int y, int flags, void* userdata)
{
	// Récupérer l'objet CDialTest à partir de userdata
	CDialTest* self = static_cast<CDialTest*>(userdata);
	int s2 = self->m_patchSize / 2;

	// Créer une copie de l'image pour dessiner le rectangle
	static cv::Mat imageCopy;
	if (imageCopy.empty()) {
		self->m_currentImage.copyTo(imageCopy);
	}

	if (event == cv::EVENT_MOUSEMOVE) {
		// Effacer l'ancien rectangle en restaurant l'image originale
		self->m_currentImage.copyTo(imageCopy);

		// Vérifier si le rectangle est dans les limites de l'image
		if (x + s2 <= self->m_currentImage.cols && y + s2 <= self->m_currentImage.rows && x > s2 && y > s2) {
			// Dessiner le rectangle en mode XOR
			cv::rectangle(imageCopy,cv::Point(x - s2, y - s2),	cv::Point(x + s2, y + s2),
				cv::Scalar(0, 0, 255), // Rouge
				1, // Épaisseur
				cv::LINE_8);
		}

		// Afficher l'image avec le rectangle
		cv::imshow("Image", imageCopy);
	}
	else if (event == cv::EVENT_LBUTTONDOWN) {
		// Clic gauche : enregistrer une imagette
		if (x + s2 <= self->m_currentImage.cols && y + s2 <= self->m_currentImage.rows && x > s2 && y > s2) {
			cv::Rect patch_rect(x - s2, y - s2, self->m_patchSize, self->m_patchSize);
			cv::Mat patch = self->m_currentImage(patch_rect);

			// Créer le chemin du fichier
			std::string class_folder = self->m_outputDirectory + "/" + std::to_string(self->m_currentClass);
			if (!std::filesystem::exists(class_folder)) {
				std::filesystem::create_directory(class_folder);
			}

			std::string patch_filename = class_folder + "/" + std::to_string(cv::getTickCount()) + ".png";
			cv::imwrite(patch_filename, patch);


		}
	}
	else if (event == cv::EVENT_RBUTTONDOWN) {
		// Clic droit : incrémenter la classe
		self->m_currentClass++;
		// Mettre à jour le titre de la fenêtre
		cv::setWindowTitle("Image", "Current class : " + std::to_string(self->m_currentClass));
	}
}


void CDialTest::OnBnClickedButtonChoiximageGt()
{
	UpdateData(true);
	m_currentClass = 0;
	m_patchSize = m_blob;
	// Étape 1 : Ouvrir une image avec CFileDialog
	CFileDialog dlgFile(TRUE, NULL, NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
		_T("Image Files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|All Files (*.*)|*.*||"), NULL);

	if (dlgFile.DoModal() != IDOK) {
		return; // L'utilisateur a annulé
	}

	CString filePath = dlgFile.GetPathName();
	std::string imagePath = CT2A(filePath);

	// Charger l'image avec OpenCV
	m_currentImage = cv::imread(imagePath);
	if (m_currentImage.empty()) {
		AfxMessageBox(_T("Erreur lors du chargement de l'image!"), MB_ICONERROR);
		return;
	}

	// Étape 2 : Choisir un dossier pour enregistrer les données
	//BROWSEINFO bi = { 0 };
	//bi.lpszTitle = _T("Choisissez un dossier pour enregistrer les patches");
	//LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
	//if (pidl != 0) {
	//	TCHAR path[MAX_PATH];
	//	if (SHGetPathFromIDList(pidl, path)) {
	//		m_outputDirectory = CT2A(path);
	//	}
	//	CoTaskMemFree(pidl);
	//}
	//else {
	//	return; // L'utilisateur a annulé
	//}
	MessageBoxA(NULL, "Selectionnez le dossier qui contiendra les sous-dossiers des classes", "Selectionner le dossier", MB_OK);
	CFolderPickerDialog folderPickerDialog(NULL, OFN_FILEMUSTEXIST, this, 0);
	if (folderPickerDialog.DoModal() != IDOK) {
		return;
	}
	CString folderPath = folderPickerDialog.GetPathName();
	std::filesystem::path rootPath(folderPath.GetString());
	m_outputDirectory = rootPath.string();

	// Initialiser la classe à 0
	m_currentClass = 0;

	// Étape 3 : Afficher l'image et gérer les clics de souris
	cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
	cv::setWindowTitle("Image", "Current class : 0 for background");
	cv::setMouseCallback("Image", CDialTest::onMouse, this);
	MessageBoxA(NULL, "Utilisez le clic gauche pour enregistrer des patches et le clic droit pour changer de classe.\nCommencez par le background", "Selectionner les patches", MB_OK);

	//std::cout << "Utilisez le clic gauche pour enregistrer des patches et le clic droit pour changer de classe." << std::endl;
	cv::imshow("Image", m_currentImage);
	//// Afficher l'image jusqu'à ce que l'utilisateur ferme la fenêtre
	//while (true) {
	//	
	//	if (cv::waitKey(10) == 27) { // Appuyer sur Échap pour quitter
	//		break;
	//	}
	//}

	//// Fermer la fenêtre
	//cv::destroyWindow("Image");
}


void CDialTest::OnBnClickedButtonCreateTrainTest()
{
	// Ouvrir une boîte de dialogue pour sélectionner le dossier source
	//BROWSEINFO bi = { 0 };
	//bi.lpszTitle = _T("Choisissez le dossier contenant les sous-dossiers des classes");
	//LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
	//if (!pidl) {
	//	AfxMessageBox(_T("Aucun dossier sélectionné."), MB_ICONERROR);
	//	return;
	//}

	//TCHAR path[MAX_PATH];
	//if (!SHGetPathFromIDList(pidl, path)) {
	//	CoTaskMemFree(pidl);
	//	AfxMessageBox(_T("Erreur lors de la sélection du dossier."), MB_ICONERROR);
	//	return;
	//}
	//CoTaskMemFree(pidl);

	//std::string sourceDir = CT2A(path);

	MessageBoxA(NULL, "Selectionnez le dossier contenant les sous-dossiers des classes", "Selectionner le dossier", MB_OK);
	CFolderPickerDialog folderPickerDialog(NULL, OFN_FILEMUSTEXIST, this, 0);
	if (folderPickerDialog.DoModal() != IDOK) {
		return;
	}
	CString folderPath = folderPickerDialog.GetPathName();
	std::filesystem::path rootPath(folderPath.GetString());
	std::string sourceDir = rootPath.string();


	// Créer les dossiers "train" et "test" dans le dossier source
	std::string trainDir = sourceDir + "/train";
	std::string testDir = sourceDir + "/test";

	try {
		fs::create_directory(trainDir);
		fs::create_directory(testDir);
	}
	catch (const std::exception& e) {
		AfxMessageBox(_T("Erreur lors de la création des dossiers train et test."), MB_ICONERROR);
		return;
	}

	// Parcourir les sous-dossiers du dossier source
	for (const auto& entry : fs::directory_iterator(sourceDir)) {
		if (entry.is_directory()) 
		{
			std::string className = entry.path().filename().string();
			if (className == "train" || className == "test") {
				continue; // Ignorer les dossiers "train" et "test"
			}
			std::string classTrainDir = trainDir + "/" + className;
			std::string classTestDir = testDir + "/" + className;

			// Créer les sous-dossiers pour chaque classe
			fs::create_directory(classTrainDir);
			fs::create_directory(classTestDir);

			// Récupérer toutes les images dans le sous-dossier de la classe
			std::vector<fs::path> images;
			for (const auto& file : fs::directory_iterator(entry.path())) {
				if (file.is_regular_file()) {
					images.push_back(file.path());
				}
			}

			// Diviser les images en deux moitiés
			size_t halfSize = images.size() / 2;

			// Copier la première moitié dans le dossier "train"
			for (size_t i = 0; i < halfSize; ++i) {
				std::string destinationPath = classTrainDir + "/" + images[i].filename().string();
				if (!fs::exists(destinationPath)) { // Vérifie si le fichier existe déjà
					fs::copy(images[i], destinationPath);
				}
			}

			// Copier la seconde moitié dans le dossier "test"
			for (size_t i = halfSize; i < images.size(); ++i) {
				std::string destinationPath = classTestDir + "/" + images[i].filename().string();
				if (!fs::exists(destinationPath)) { // Vérifie si le fichier existe déjà
					fs::copy(images[i], destinationPath);
				}
			}
		}
	}

	AfxMessageBox(_T("Les dossiers train et test ont été créés avec succès."), MB_ICONINFORMATION);
}


//void CDialTest::OnBnClickedButtonCreateTrainTest()
//{
//	//// TODO: ajoutez ici le code de votre gestionnaire de notification de contrôle
//	//// lire le fichier "fenetres.txt" et appeler le décodage braille
//	//std::ifstream file("fenetres.csv");
//	//if (!file.is_open()) {
//	//	AfxMessageBox(_T("Erreur lors de l'ouverture du fichier fenetres.csv!"), MB_ICONERROR);
//	//	return;
//	//}
//
//	//std::vector<BraillePoint> win;
//	//std::string line;
//	//bool is_header = true;
//
//	//int taille_v = 32;
//	//int taille_h = 32;
//
//	//// Lire le fichier ligne par ligne
//	//while (std::getline(file, line)) {
//	//	if (is_header) {
//	//		// Ignorer la première ligne (en-tête)
//	//		is_header = false;
//	//		continue;
//	//	}
//
//	//	std::istringstream iss(line);
//	//	std::string token;
//	//	int class_index;
//	//	float prob;
//	//	cv::Rect rect;
//
//	//	// Lire les valeurs séparées par des virgules
//	//	if (std::getline(iss, token, ',')) class_index = std::stoi(token);
//	//	if (std::getline(iss, token, ',')) prob = std::stof(token);
//	//	if (std::getline(iss, token, ',')) rect.x = std::stoi(token);
//	//	if (std::getline(iss, token, ',')) rect.y = std::stoi(token);
//	//	if (std::getline(iss, token, ',')) rect.width = std::stoi(token);
//	//	if (std::getline(iss, token, ',')) rect.height = std::stoi(token);
//
//	//	// Ajouter le point Braille correspondant
//	//	win.push_back({ (float)(rect.x + rect.width / 2), (float)(rect.y + rect.height / 2) });
//	//}
//
//	//file.close();
//
//
//
//
//	//// Charger l'image avec OpenCV
//	//cv::Mat image = cv::imread(m_image_path);
//	//if (image.empty()) {
//	//	// Afficher la boîte de dialogue pour sélectionner une image
//	//	CFileDialog dlgFile(TRUE, NULL, NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
//	//		_T("Image Files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png|All Files (*.*)|*.*||"), NULL);
//
//	//	if (dlgFile.DoModal() != IDOK) {
//	//		return; // L'utilisateur a annulé
//	//	}
//
//	//	// Récupérer le chemin de l'image sélectionnée
//	//	CString filePath = dlgFile.GetPathName();
//	//	std::string imagePath = CT2A(filePath);
//	//	// Charger l'image avec OpenCV
//	//	image = cv::imread(imagePath);
//	//	m_image_path = imagePath;
//	//}
//	////// Créer une instance du décodeur
//	////BrailleRecognizer recognizer(m_pMyNet->m_input_width, m_pMyNet->m_input_height);
//
//	////std::string recognizedText = recognizer.recognizeText(win);
//
//	//CBrailleMusicDecoder decoder(win, 2* taille_h,3* taille_v);
//	////std::string recognizedText = decoder.decode(false);
//	////recognizer.overlayCharactersOnImage(image, cv::Scalar(0, 255, 0));
//	//// Afficher le texte reconnu
//	//decoder.overlayDecodedCharacters(image, cv::Scalar(0, 0, 0),1,2);
//	//// Afficher l'image avec les caractères incrustés
//	//cv::imshow("Image avec Braille", image);
//
//	//// Sauvegarder l'image modifiée
//	//cv::imwrite("image_with_braille.png", image);
//
//	//// Attendre une touche pour fermer la fenêtre
//	//cv::waitKey(0);
//
//
//	// créer un dossier pour les images de test
//
//}

BOOL CDialTest::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	m_blob = m_pMyNet->m_input_width;
	UpdateData(false);
	// TODO:  Ajoutez ici une initialisation supplémentaire

	return TRUE;  // return TRUE unless you set the focus to a control
	// EXCEPTION : les pages de propriétés OCX devraient retourner FALSE
}
