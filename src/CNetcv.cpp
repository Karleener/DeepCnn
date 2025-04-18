#include "pch.h"
#include "CNetcv.h"
#include "courbe.h"
#include <torch/data/datasets/tensor.h>
#include <torch/data/transforms.h>
#include <torch/data/dataloader.h>
#include <torch/data/samplers.h>
#include <Windows.h>
#include <sstream>
#include <string>

using namespace cv;
using namespace torch;
using namespace std;

class CustomTensorDataset : public torch::data::Dataset<CustomTensorDataset> {
private:
	torch::Tensor images_;
	torch::Tensor labels_;

public:
	CustomTensorDataset(torch::Tensor images, torch::Tensor labels)
		: images_(images), labels_(labels) {
	}

	torch::data::Example<> get(size_t index) override {
		return { images_[index], labels_[index] };
	}

	torch::optional<size_t> size() const override {
		return images_.size(0);
	}
};



CNetcv::CNetcv()
{
	m_Accuracy = 0;
	m_Loss = 0;
}

//void CNetcv::loadDataset(const string& datasetPath, std::vector<cv::Mat>& images, std::vector<int>& labels)
//{
//
//	int label = 0;
//	class_names.clear();
//	for (const auto& entry : filesystem::directory_iterator(datasetPath))
//	{
//		if (entry.is_directory())
//		{
//			//int label = stoi(entry.path().filename().string());
//			for (const auto& file : filesystem::directory_iterator(entry.path())) {
//				if (file.is_regular_file()) 
//				{
//					Mat img = imread(file.path().string()/*, IMREAD_GRAYSCALE*/);
//					if (!img.empty()) 
//					{
//						images.push_back(img);
//						labels.push_back(label);
//						
//					}
//				}
//			}
//			class_names.push_back(entry.path().filename().string());
//			label++;
//		}
//	}
//	m_nb_classes = label;
//
//	int num_samples = images.size();
//	m_image_tensor = torch::empty({ num_samples, 3, 28, 28 }, torch::kFloat32);
//	m_label_tensor = torch::empty({ num_samples }, torch::kLong);
//
//	for (int i = 0; i < num_samples; ++i)
//	{
//		cv::Mat img_resized;
//		cv::resize(images[i], img_resized, cv::Size(28, 28));
//
//		// Convertir Mat en tensor
//		torch::Tensor img_tensor = torch::from_blob(img_resized.data,	{ 1, img_resized.rows, img_resized.cols, 3 },	torch::kByte	);
//
//		// Permuter les dimensions et normaliser
//		img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32).div(255.0);
//
//		m_image_tensor[i] = img_tensor[0];
//		m_label_tensor[i] = labels[i];
//	}
//
//}

bool CNetcv::loadDataset(const string& datasetPath)
{
	class_names.clear();
	m_train_image_paths.clear();
	m_train_labels.clear();
	m_test_image_paths.clear();
	m_test_labels.clear();

	std::vector<std::string> subdirs = { "train", "test" };
	std::vector<std::vector<std::string>*> image_paths = { &m_train_image_paths, &m_test_image_paths };
	std::vector<std::vector<int>*> labels = { &m_train_labels, &m_test_labels };

	for (size_t i = 0; i < subdirs.size(); ++i) {
		int label = 0;
		std::string subdir_path = datasetPath + "/" + subdirs[i];
		if (!filesystem::exists(subdir_path)) {
			std::cerr << "Le répertoire " << subdir_path << " n'existe pas." << std::endl;
			return false;
		}
		for (const auto& entry : filesystem::directory_iterator(subdir_path)) {
			if (entry.is_directory()) {
				for (const auto& file : filesystem::directory_iterator(entry.path())) {
					if (file.is_regular_file()) {
						image_paths[i]->push_back(file.path().string());
						labels[i]->push_back(label);
					}
				}
				if (i == 0) { // Ajouter les noms de classes uniquement pour l'ensemble d'entraînement
					class_names.push_back(entry.path().filename().string());
				}
				label++;
			}
		}
	}

	if (m_train_image_paths.empty() || m_test_image_paths.empty()) {
		return false;
	}

	m_nb_classes = class_names.size();
	return true;
}

//bool CNetcv::loadDataset(const string& datasetPath, std::vector<cv::Mat>& train_images, std::vector<int>& train_labels, std::vector<cv::Mat>& test_images, std::vector<int>& test_labels)
//{
//	class_names.clear();
//	std::vector<std::string> subdirs = { "train", "test" };
//	std::vector<std::vector<cv::Mat>*> images = { &train_images, &test_images };
//	std::vector<std::vector<int>*> labels = { &train_labels, &test_labels };
//
//	for (size_t i = 0; i < subdirs.size(); ++i) {
//		int label = 0;
//		std::string subdir_path = datasetPath + "/" + subdirs[i];
//		if (!filesystem::exists(subdir_path)) {
//			std::cerr << "Le répertoire " << subdir_path << " n'existe pas." << std::endl;
//			return false;
//		}
//		for (const auto& entry : filesystem::directory_iterator(subdir_path)) {
//			if (entry.is_directory()) {
//				for (const auto& file : filesystem::directory_iterator(entry.path())) {
//					if (file.is_regular_file()) {
//						Mat img = imread(file.path().string());
//						if (!img.empty()) {
//							images[i]->push_back(img);
//							labels[i]->push_back(label);
//						}
//					}
//				}
//				if (i == 0) { // Ajouter les noms de classes uniquement pour l'ensemble d'entraînement
//					class_names.push_back(entry.path().filename().string());
//				}
//				label++;
//			}
//		}
//	}
//	if (train_images.size() == 0 || test_images.size() == 0)
//	{
//		return false;
//	}
//	m_nb_classes = class_names.size();
//
//	// Créer les tenseurs pour l'ensemble d'entraînement
//	int num_train_samples = train_images.size();
//	m_train_image_tensor = torch::empty({ num_train_samples, 3, m_input_height, m_input_width }, torch::kFloat32);
//	m_train_label_tensor = torch::empty({ num_train_samples }, torch::kLong);
//
//	for (int i = 0; i < num_train_samples; ++i) {
//		cv::Mat img_resized;
//		cv::resize(train_images[i], img_resized, cv::Size(m_input_height, m_input_width));
//		torch::Tensor img_tensor = torch::from_blob(img_resized.data, { 1, img_resized.rows, img_resized.cols, 3 }, torch::kByte);
//		img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32).div(255.0);
//		m_train_image_tensor[i] = img_tensor[0];
//		m_train_label_tensor[i] = train_labels[i];
//	}
//
//	// Créer les tenseurs pour l'ensemble de test
//	int num_test_samples = test_images.size();
//	m_test_image_tensor = torch::empty({ num_test_samples, 3, m_input_height, m_input_width }, torch::kFloat32);
//	m_test_label_tensor = torch::empty({ num_test_samples }, torch::kLong);
//
//	for (int i = 0; i < num_test_samples; ++i) {
//		cv::Mat img_resized;
//		cv::resize(test_images[i], img_resized, cv::Size(m_input_height, m_input_width));
//		torch::Tensor img_tensor = torch::from_blob(img_resized.data, { 1, img_resized.rows, img_resized.cols, 3 }, torch::kByte);
//		img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32).div(255.0);
//		m_test_image_tensor[i] = img_tensor[0];
//		m_test_label_tensor[i] = test_labels[i];
//	}
//	return true;
//}

void CNetcv::testInference( const vector<Mat>& testImages)
{

}

void CNetcv::trainModel(Tensor& image_tensor,Tensor& label_tensor,	int NbClasses, int num_epochs = 10)
{
	// Vérifier si CUDA est disponible
	torch::DeviceType device_type = torch::cuda::is_available()
		? torch::kCUDA
		: torch::kCPU;
	torch::Device device(device_type);

	// Créer le modèle
	int num_classes = NbClasses;  // Adaptez selon votre dataset
	auto model = std::make_shared<CNNModel>(num_classes);
	model->to(device);

	// Charger les données
	//auto [image_tensor, label_tensor] = loadDataset(images, labels);
	image_tensor = image_tensor.to(device);
	label_tensor = label_tensor.to(device);

	// Optimiseur et fonction de perte
	torch::optim::Adam optimizer(model->parameters(), 0.001);
	auto criterion = torch::nn::NLLLoss();


	CCourbe* MaCourbe = new CCourbe;
	MaCourbe->Create(NULL, L"Loss (blue) and accuracy (red)",WS_POPUPWINDOW|WS_OVERLAPPEDWINDOW,CRect(0,0,400,400));
	MaCourbe->ShowWindow(TRUE);

	vector<double> data_x(num_epochs);
	vector<double> data_y(num_epochs); // loss
	vector<double> data_z(num_epochs); // accuracy
	for (int i = 0; i < num_epochs; i++)
	{
		data_x[i] = double(i); 
		data_y[i] = 0.0;
		data_z[i] = 0.0;
	} 
	MaCourbe->Dessine(data_x, data_y, data_z,0);

	// Boucle d'entraînement
	for (int epoch = 0; epoch < num_epochs; ++epoch)
	{
		// Mode entraînement
		model->train();

		// Réinitialiser les gradients
		optimizer.zero_grad();

		// Propagation avant
		auto output = model->forward(image_tensor);
		auto loss = criterion(output, label_tensor);

		auto predictions = torch::argmax(output, 1);
		auto correct = (predictions == label_tensor).sum();
		float accuracy = correct.item<float>() / label_tensor.size(0);

		// Rétropropagation
		loss.backward();

		// Mise à jour des poids
		optimizer.step();

		// Afficher la progression dans une MessageBox
		//std::ostringstream oss;
		//oss << "Epoch [" << epoch + 1 << "/" << num_epochs << "] Loss: " << loss.item<float>();
		//std::string message = oss.str();
		//MessageBoxA(NULL, message.c_str(), "Training Progress", MB_OK);

		data_y[epoch] = loss.item<float>();
		data_z[epoch] = accuracy;

		MaCourbe->Dessine(data_x, data_y, data_z,epoch);

	}
}

//void CNetcv::trainModelDyn(int num_epochs, double lr)
//{
//	// Vérifier si CUDA est disponible
//	torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//	torch::Device device(device_type);
//
//	// Créer le modèle
//	int num_classes = m_nb_classes;
//	m_model->to(device);
//
//	// Déplacer les données sur le périphérique approprié
//	m_train_image_tensor = m_train_image_tensor.to(device);
//	m_train_label_tensor = m_train_label_tensor.to(device);
//	m_test_image_tensor = m_test_image_tensor.to(device);
//	m_test_label_tensor = m_test_label_tensor.to(device);
//
//	// Optimiseur et fonction de perte
//	torch::optim::Adam optimizer(m_model->parameters(), lr);
//	auto criterion = torch::nn::NLLLoss();
//
//	// Initialiser les courbes
//	CCourbe* MaCourbeTrain = new CCourbe;
//	MaCourbeTrain->Create(NULL, L"Loss (blue) and accuracy (red) Train", WS_POPUPWINDOW | WS_OVERLAPPEDWINDOW, CRect(0, 0, 400, 400));
//	MaCourbeTrain->ShowWindow(TRUE);
//
//	CCourbe* MaCourbeTest = new CCourbe;
//	MaCourbeTest->Create(NULL, L"Loss (blue) and accuracy (red) Test", WS_POPUPWINDOW | WS_OVERLAPPEDWINDOW, CRect(0, 420, 400, 820));
//	MaCourbeTest->ShowWindow(TRUE);
//
//	vector<double> data_x(num_epochs);
//	vector<double> train_loss(num_epochs);
//	vector<double> train_accuracy(num_epochs);
//	vector<double> test_loss(num_epochs);
//	vector<double> test_accuracy(num_epochs);
//
//	for (int i = 0; i < num_epochs; i++) {
//		data_x[i] = double(i);
//		train_loss[i] = 0.0;
//		train_accuracy[i] = 0.0;
//		test_loss[i] = 0.0;
//		test_accuracy[i] = 0.0;
//	}
//	MaCourbeTrain->Dessine(data_x, train_loss, train_accuracy, 0);
//	MaCourbeTest->Dessine(data_x, test_loss, test_accuracy, 0);
//
//	// Boucle d'entraînement
//	for (int epoch = 0; epoch < num_epochs; ++epoch) {
//		// Mode entraînement
//		m_model->train();
//		optimizer.zero_grad();
//		auto output = m_model->forward(m_train_image_tensor);
//		auto loss = criterion(output, m_train_label_tensor);
//		loss.backward();
//		optimizer.step();
//
//		// Calculer la précision d'entraînement
//		auto predictions = torch::argmax(output, 1);
//		auto correct = (predictions == m_train_label_tensor).sum();
//		float accuracy = correct.item<float>() / m_train_label_tensor.size(0);
//
//		train_loss[epoch] = loss.item<float>();
//		train_accuracy[epoch] = accuracy;
//
//		// Mode évaluation
//		m_model->eval();
//		torch::NoGradGuard no_grad;
//		auto test_output = m_model->forward(m_test_image_tensor);
//		auto test_loss_value = criterion(test_output, m_test_label_tensor);
//		auto test_predictions = torch::argmax(test_output, 1);
//		auto test_correct = (test_predictions == m_test_label_tensor).sum();
//		float test_accuracy_value = test_correct.item<float>() / m_test_label_tensor.size(0);
//
//		test_loss[epoch] = test_loss_value.item<float>();
//		test_accuracy[epoch] = test_accuracy_value;
//
//		// Mettre à jour les courbes
//		MaCourbeTrain->Dessine(data_x, train_loss, train_accuracy, epoch);
//		MaCourbeTest->Dessine(data_x, test_loss, test_accuracy, epoch);
//	}
//}


void CNetcv::trainModelDyn(int num_epochs, double lr, int batch_size, int Periode, string nomModel)
{
	// Vérifier si CUDA est disponible
	torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	//device_type = torch::kCPU;
	torch::Device device(device_type);

	// Créer le modèle
	int num_classes = m_nb_classes;
	m_model->to(device);

	// Créer le DataLoader pour les données d'entraînement
	auto train_dataset = CustomImageDataset(m_train_image_paths, m_train_labels, m_input_height, m_input_width)
		.map(torch::data::transforms::Stack<>());
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(0));

	// Créer le DataLoader pour les données de test
	auto test_dataset = CustomImageDataset(m_test_image_paths, m_test_labels, m_input_height, m_input_width)
		.map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(0));

	// Optimiseur et fonction de perte
	torch::optim::Adam optimizer(m_model->parameters(), lr);
	auto criterion = torch::nn::NLLLoss();

	// Initialiser les courbes
	CCourbe* MaCourbeTrain = new CCourbe;
	MaCourbeTrain->Create(NULL, L"Loss (blue) and accuracy (red) Train", WS_POPUPWINDOW | WS_OVERLAPPEDWINDOW, CRect(0, 0, 600, 400));
	MaCourbeTrain->ShowWindow(TRUE);

	CCourbe* MaCourbeTest = new CCourbe;
	MaCourbeTest->Create(NULL, L"Loss (blue) and accuracy (red) Test", WS_POPUPWINDOW | WS_OVERLAPPEDWINDOW, CRect(0, 420, 600, 820));
	MaCourbeTest->ShowWindow(TRUE);

	AfxGetMainWnd()->SetFocus();

	vector<double> data_x(num_epochs);
	vector<double> train_loss(num_epochs);
	vector<double> train_accuracy(num_epochs);
	vector<double> test_loss(num_epochs);
	vector<double> test_accuracy(num_epochs);

	for (int i = 0; i < num_epochs; i++) {
		data_x[i] = double(i);
		train_loss[i] = 0.0;
		train_accuracy[i] = 0.0;
		test_loss[i] = 0.0;
		test_accuracy[i] = 0.0;
	}
	//MaCourbeTrain->Dessine(data_x, train_loss, train_accuracy, 0);
	//MaCourbeTest->Dessine(data_x, test_loss, test_accuracy, 0);

	// Boucle d'entraînement
	for (int epoch = 0; epoch < num_epochs; ++epoch) {
		if (GetAsyncKeyState(VK_F2) & 0x8000) {
			AfxMessageBox(_T("Entraînement interrompu par l'utilisateur."), MB_ICONINFORMATION);
			break;
		}

		// Mode entraînement
		m_model->train();
		double epoch_loss = 0.0;
		int correct = 0;
		int total = 0;

		for (auto& batch : *train_loader) {
			auto images = batch.data.to(device);
			auto labels = batch.target.to(device);

			optimizer.zero_grad();
			auto output = m_model->forward(images);
			auto loss = criterion(output, labels);
			loss.backward();
			optimizer.step();

			epoch_loss += loss.item<float>() * images.size(0);
			auto predictions = torch::argmax(output, 1);
			correct += (predictions == labels).sum().item<int>();
			total += labels.size(0);
		}

		float accuracy = static_cast<float>(correct) / total;
		train_loss[epoch] = epoch_loss / total;
		train_accuracy[epoch] = accuracy;

		// Mode évaluation
		m_model->eval();
		torch::NoGradGuard no_grad;
		double test_epoch_loss = 0.0;
		int test_correct = 0;
		int test_total = 0;

		for (auto& batch : *test_loader) {
			auto images = batch.data.to(device);
			auto labels = batch.target.to(device);

			auto output = m_model->forward(images);
			auto loss = criterion(output, labels);

			test_epoch_loss += loss.item<float>() * images.size(0);
			auto predictions = torch::argmax(output, 1);
			test_correct += (predictions == labels).sum().item<int>();
			test_total += labels.size(0);
		}

		float test_accuracy_value = static_cast<float>(test_correct) / test_total;
		test_loss[epoch] = test_epoch_loss / test_total;
		test_accuracy[epoch] = test_accuracy_value;

		// Mettre à jour les courbes
		if (epoch > 0)
		{
		
			MSG msg;
			while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

		MaCourbeTrain->Dessine(data_x, train_loss, train_accuracy, epoch);
		MaCourbeTest->Dessine(data_x, test_loss, test_accuracy, epoch);

	//	MaCourbeTrain->Invalidate();
	//	MaCourbeTest->Invalidate();
		}



		// Mettre à jour les fenêtres avec les valeurs actuelles
		CString str;
		str.Format(L"Train : Epoch [%d/%d] - Loss: %.4f - Accuracy: %.2f%%", epoch + 1, num_epochs, train_loss[epoch], train_accuracy[epoch] * 100);
		MaCourbeTrain->SetWindowText(str);
		str.Format(L"Test : Epoch [%d/%d] - Loss: %.4f - Accuracy: %.2f%%", epoch + 1, num_epochs, test_loss[epoch], test_accuracy[epoch] * 100);
		MaCourbeTest->SetWindowText(str);

		// Sauvegarder le modèle toutes les `Periode` epochs
		if (epoch % Periode == 0) {
			std::string modelPath = nomModel.empty() ? "model_epoch_" + std::to_string(epoch) : nomModel + "_epoch_" + std::to_string(epoch);
			saveModel(m_model, modelPath, "");
		}
	}

	m_Accuracy = test_accuracy[num_epochs - 1];
	m_Loss = test_loss[num_epochs - 1];

	MessageBox(NULL, L"Apprentissage terminé", L"Succès", MB_OK);
	MaCourbeTest->DestroyWindow();
	MaCourbeTrain->DestroyWindow();

}

//void CNetcv::trainModelDyn(int num_epochs, double lr, int batch_size, int Periode, string nomModel)
//{
//
//
//
//	// Dans votre fonction principale ou là où vous voulez afficher l'information
//	//std::ostringstream oss;
//	//oss << "Chemin DLL torch: " << (void*)&torch::cuda::is_available;
//	//std::string message = oss.str();
//	//MessageBoxA(NULL, message.c_str(), "Informations LibTorch", MB_OK | MB_ICONINFORMATION);
//	//// Vérifier si CUDA est disponible
//	//LoadLibraryA("torch_cuda.dll");
//
//	//try {
//	//	// Tente de forcer l'initialisation CUDA
//	//	torch::Tensor test = torch::ones({ 1, 1 }, torch::Device(torch::kCUDA));
//	//	std::string msg = "Initialisation CUDA réussie!";
//	//	MessageBoxA(NULL, msg.c_str(), "CUDA Test", MB_OK);
//	//}
//	//catch (const c10::Error& e) {
//	//	std::string msg = "Erreur CUDA: ";
//	//	msg += e.what();
//	//	MessageBoxA(NULL, msg.c_str(), "CUDA Error", MB_OK | MB_ICONERROR);
//	//}
//
//
//
//
//	bool testcuda = torch::cuda::is_available();
//	torch::DeviceType device_type;
//	if (testcuda) {
//		device_type = torch::kCUDA;
//	}
//	else {
//		device_type = torch::kCPU;
//	}
//	//torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//
//	torch::Device device(device_type);
//
//	// Créer le modèle
//	int num_classes = m_nb_classes;
//	m_model->to(device);
//
//	// Déplacer les données sur le périphérique approprié
//	m_train_image_tensor = m_train_image_tensor.to(device);
//	m_train_label_tensor = m_train_label_tensor.to(device);
//	m_test_image_tensor = m_test_image_tensor.to(device);
//	m_test_label_tensor = m_test_label_tensor.to(device);
//
//	// Créer un DataLoader pour les données d'entraînement
//	//auto train_dataset = torch::data::datasets::TensorDataset(m_train_image_tensor).map(torch::data::transforms::Stack<>());
//
//	auto train_dataset = CustomTensorDataset(m_train_image_tensor, m_train_label_tensor).map(torch::data::transforms::Stack<>());
//	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);
//
//	// Optimiseur et fonction de perte
//	torch::optim::Adam optimizer(m_model->parameters(), lr);
//	auto criterion = torch::nn::NLLLoss();
//
//	// Initialiser les courbes
//	CCourbe* MaCourbeTrain = new CCourbe;
//	MaCourbeTrain->Create(NULL, L"Loss (blue) and accuracy (red) Train", WS_POPUPWINDOW | WS_OVERLAPPEDWINDOW, CRect(0, 0, 600,400));
//	MaCourbeTrain->ShowWindow(TRUE);
//
//	CCourbe* MaCourbeTest = new CCourbe;
//	MaCourbeTest->Create(NULL, L"Loss (blue) and accuracy (red) Test", WS_POPUPWINDOW | WS_OVERLAPPEDWINDOW, CRect(0, 420, 600, 820));
//	MaCourbeTest->ShowWindow(TRUE);
//
//	AfxGetMainWnd()->SetFocus();
//
//	vector<double> data_x(num_epochs);
//	vector<double> train_loss(num_epochs);
//	vector<double> train_accuracy(num_epochs);
//	vector<double> test_loss(num_epochs);
//	vector<double> test_accuracy(num_epochs);
//
//	for (int i = 0; i < num_epochs; i++) {
//		data_x[i] = double(i);
//		train_loss[i] = 0.0;
//		train_accuracy[i] = 0.0;
//		test_loss[i] = 0.0;
//		test_accuracy[i] = 0.0;
//	}
//	MaCourbeTrain->Dessine(data_x, train_loss, train_accuracy, 0);
//	MaCourbeTest->Dessine(data_x, test_loss, test_accuracy, 0);
//	int key = 0;
//	bool stop = false;
//	// Boucle d'entraînement
//	for (int epoch = 0; epoch < num_epochs; ++epoch) 
//	{
//		//MSG msg;
//		//while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
//		//	if (msg.message == WM_QUIT) {
//		//		stop = true;
//		//	}
//		//	TranslateMessage(&msg);
//		//	DispatchMessage(&msg);
//		//}
//
//		//if (stop) {
//		//	break;
//		//}
//		if (GetAsyncKeyState(VK_F5) & 0x8000) 
//		{
//			AfxMessageBox(_T("Entraînement interrompu par l'utilisateur."), MB_ICONINFORMATION);
//			break;
//		}
//		//key=cv::waitKey(1);
//		//if (key == 27) // Échapper
//		//{
//		//	AfxMessageBox(_T("Entraînement interrompu par l'utilisateur."), MB_ICONINFORMATION);
//		//	break;
//		//}
//		// Mode entraînement
//		m_model->train();
//		double epoch_loss = 0.0;
//		int correct = 0;
//		int total = 0;
//
//		for (auto& batch : *train_loader) {
//			auto images = batch.data.to(device);
//			auto labels = batch.target.to(device);
//
//			optimizer.zero_grad();
//			auto output = m_model->forward(images);
//			auto loss = criterion(output, labels);
//			loss.backward();
//			optimizer.step();
//
//			epoch_loss += loss.item<float>() * images.size(0);
//			auto predictions = torch::argmax(output, 1);
//			correct += (predictions == labels).sum().item<int>();
//			total += labels.size(0);
//		}
//
//		float accuracy = static_cast<float>(correct) / total;
//		train_loss[epoch] = epoch_loss / total;
//		train_accuracy[epoch] = accuracy;
//
//		// Mode évaluation
//		m_model->eval();
//		torch::NoGradGuard no_grad;
//		double test_epoch_loss = 0.0;
//		int test_correct = 0;
//		int test_total = 0;
//
//		auto test_dataset = CustomTensorDataset(m_test_image_tensor, m_test_label_tensor).map(torch::data::transforms::Stack<>());
//		auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);
//
//		for (auto& batch : *test_loader) {
//			auto images = batch.data.to(device);
//			auto labels = batch.target.to(device);
//
//			auto output = m_model->forward(images);
//			auto loss = criterion(output, labels);
//
//			test_epoch_loss += loss.item<float>() * images.size(0);
//			auto predictions = torch::argmax(output, 1);
//			test_correct += (predictions == labels).sum().item<int>();
//			test_total += labels.size(0);
//		}
//
//		float test_accuracy_value = static_cast<float>(test_correct) / test_total;
//		test_loss[epoch] = test_epoch_loss / test_total;
//		test_accuracy[epoch] = test_accuracy_value;
//
//		// Mettre à jour les courbes
//		MaCourbeTrain->Dessine(data_x, train_loss, train_accuracy, epoch);
//		MaCourbeTest->Dessine(data_x, test_loss, test_accuracy, epoch);
//
//		MaCourbeTrain->Invalidate();
//		MaCourbeTest->Invalidate();
//		// mettre accuracy dans une CString
//		CString str;
//		str.Format(L"Train : Epoch [%d/%d] - Loss: %.4f - Accuracy: %.2f%%", epoch + 1, num_epochs, train_loss[epoch], train_accuracy[epoch] * 100);
//		MaCourbeTrain->SetWindowText(str);
//		str.Format(L"Test : Epoch [%d/%d] - Loss: %.4f - Accuracy: %.2f%%", epoch + 1, num_epochs, test_loss[epoch], test_accuracy[epoch] * 100);
//		MaCourbeTest->SetWindowText(str);
//		// sauvegarder le modèle toutes les Periode epochs avec un nom qui contient le numéro d'époque
//		if (epoch % Periode == 0) {
//			if (nomModel != "") {
//				std::string modelPath = nomModel + "_epoch_" + std::to_string(epoch) ;
//				saveModel(m_model, modelPath, "");
//			}
//			else
//			{
//				std::string modelPath = "model_epoch_" + std::to_string(epoch) ;
//				saveModel(m_model, modelPath, "");
//			}
//	
//		}
//
//		
//	}
//	m_Accuracy = test_accuracy[num_epochs - 1];
//	m_Loss = test_loss[num_epochs - 1];
//
//	MessageBox(NULL,L"Apprentissage termine", L"Succes", MB_OK);
//	MaCourbeTest->DestroyWindow();
//	MaCourbeTrain->DestroyWindow();
//
//}

bool CNetcv::saveModel(std::shared_ptr<CNNModel> model, const std::string& filePath) 
{
	try {
		torch::save(model, filePath);
		std::cout << "Modèle sauvegardé avec succès dans : " << filePath << std::endl;
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Erreur lors de la sauvegarde du modèle : " << e.what() << std::endl;
		return false;
	}
}

// Fonction pour charger le modèle
std::shared_ptr<CNNModel> CNetcv::loadModel(const std::string& filePath, int numClasses)
{
	try {
		// Créer une instance de modèle vide
		std::shared_ptr<CNNModel> model = std::make_shared<CNNModel>(numClasses);

		// Charger les paramètres depuis le fichier
		torch::load(model, filePath);

		// Vérifier si CUDA est disponible et déplacer le modèle sur le périphérique approprié
		torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		torch::Device device(device_type);
		model->to(device);

		std::cout << "Modèle chargé avec succès depuis : " << filePath << std::endl;
		return model;
	}
	catch (const std::exception& e) {
		std::cerr << "Erreur lors du chargement du modèle : " << e.what() << std::endl;
		return nullptr;
	}
}

std::shared_ptr<CNNModelDyn> CNetcv::createModel(const std::vector<ConvLayerParams>& conv_params, const std::vector<DenseLayerParams>& dense_params, int input_height, int input_width, int num_classes)
{
	m_model = std::make_shared<CNNModelDyn>(conv_params, dense_params, input_height, input_width, num_classes);
	m_input_height = input_height;
	m_input_width = input_width;
	// Enregistrer les paramètres de convolution
	m_conv_params = conv_params;
	return m_model;
}

// Fonction pour sauvegarder le modèle paramétrable
bool CNetcv::saveModel(std::shared_ptr<CNNModelDyn> model, const std::string& filePath, const std::string& namesFilePath)
{
	try {
		// 1. Sauvegarder la configuration du modèle
		std::string configPath = filePath + ".configCNN";
		// enregistrer le fichier de configuration
		// 
		if (namesFilePath != "") saveConfig(m_conv_params, m_dens_params, configPath);

		//std::ofstream configFile(configPath);

		//if (!configFile.is_open()) {
		//	std::cerr << "Impossible d'ouvrir le fichier de configuration : " << configPath << std::endl;
		//	return false;
		//}

		//// Écrire les dimensions d'entrée et le nombre de classes
		//configFile << m_input_height << " " << m_input_width << " " << m_nb_classes << std::endl;

		//// Écrire le nombre de couches de convolution
		//configFile << m_conv_params.size() << std::endl;

		//// Écrire les paramètres de chaque couche
		//for (const auto& params : m_conv_params) {
		//	configFile << params.in_channels << " "
		//		<< params.out_channels << " "
		//		<< params.kernel_size << " "
		//		<< params.stride << " "
		//		<< params.padding << " "
		//		<< (params.use_pool ? 1 : 0) << " "
		//		<< params.pool_size << std::endl;
		//}

		//configFile.close();

		// 2. Sauvegarder les poids du modèle
		torch::save(model, filePath+".pt");

		// Sauvegarder les noms des classes
		//std::string namesFilePath = filePath.substr(0, filePath.find_last_of('.')) + ".names";
		
		if (namesFilePath != "")
		{
			std::ofstream namesFile(namesFilePath);
			if (!namesFile.is_open()) {
				std::cerr << "Erreur lors de l'ouverture du fichier pour sauvegarder les noms des classes." << std::endl;
				return false;
			}

			for (const auto& className : class_names) {
				namesFile << className << std::endl;
			}

			namesFile.close();
		}

		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Erreur lors de la sauvegarde du modèle : " << e.what() << std::endl;
		return false;
	}
}

// Fonction pour charger le modèle paramétrable
std::shared_ptr<CNNModelDyn> CNetcv::loadModel(const std::string& filePath) {
	try {
		// 1. Charger la configuration du modèle
		std::string configPath = filePath + ".configCNN";
		std::ifstream configFile(configPath);

		if (!configFile.is_open()) {
			std::cerr << "Impossible d'ouvrir le fichier de configuration : " << configPath << std::endl;
			return nullptr;
		}

		// Lire les dimensions d'entrée et le nombre de classes
		int input_height, input_width, num_classes;
		configFile >> input_height >> input_width >> num_classes;

		// Lire le nombre de couches de convolution
		size_t num_layers;
		configFile >> num_layers;

		// Lire les paramètres de chaque couche
		std::vector<ConvLayerParams> conv_params;
		for (size_t i = 0; i < num_layers; ++i) {
			ConvLayerParams params;
			int use_pool_int;

			configFile >> params.in_channels
				>> params.out_channels
				>> params.kernel_size
				>> params.stride
				>> params.padding
				>> use_pool_int
				>> params.pool_size;

			params.use_pool = (use_pool_int == 1);
			conv_params.push_back(params);
		}
		std::vector<DenseLayerParams> denseLayers;
		size_t denseLayerCount;
		configFile >> denseLayerCount;
		denseLayers.resize(denseLayerCount);
		for (size_t i = 0; i < denseLayerCount; ++i)
		{
			configFile >> denseLayers[i].nb_neurons >> denseLayers[i].activation_type;
		}

		configFile.close();

		// 2. Créer le modèle avec la configuration chargée
		m_model = std::make_shared<CNNModelDyn>(conv_params, denseLayers, input_height, input_width, num_classes);

		// 3. Charger les poids du modèle
		torch::load(m_model, filePath+".pt");

		// 4. Déplacer le modèle sur le périphérique approprié
		torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		torch::Device device(device_type);
		m_model->to(device);
		std::cout << "Modèle chargé avec succès depuis : " << filePath << std::endl;

		// Charger les noms des classes
		std::string namesFilePath = filePath.substr(0, filePath.find_last_of('.')) + ".names";
		std::ifstream namesFile(namesFilePath);
		if (!namesFile.is_open()) {
			std::cerr << "Erreur lors de l'ouverture du fichier pour charger les noms des classes." << std::endl;
			return m_model;
		}

		class_names.clear();
		std::string className;
		while (std::getline(namesFile, className)) {
			class_names.push_back(className);
		}

		namesFile.close();

		return m_model;
	}
	catch (const std::exception& e) {
		std::cerr << "Erreur lors du chargement du modèle : " << e.what() << std::endl;
		return nullptr;
	}
}


std::map<std::string, float>  CNetcv::testImageInference(std::shared_ptr<CNNModelDyn>& model, Mat& input_image)
{
	std::map<std::string, float> class_probabilities;
	try {
		// 1. Vérifier que le modèle existe
		if (!model) {
			AfxMessageBox(CString("Erreur: modèle non chargé"), MB_OK);
			return class_probabilities;
		}

		// 2. Vérifier que l'image n'est pas vide
		if (input_image.empty()) {
			AfxMessageBox(CString("Erreur: image vide"), MB_OK);
			return class_probabilities;
		}

		// 3. Vérifier que l'image a le bon nombre de canaux (3 pour RGB)
		if (input_image.channels() != 3) {
			std::string msg = "Erreur: l'image doit avoir 3 canaux (RGB), nombre actuel: " +
				std::to_string(input_image.channels());
			AfxMessageBox(CString(msg.c_str()), MB_OK);
			return class_probabilities;
		}

		// 4. Vérifier et redimensionner l'image si nécessaire
		cv::Mat resized_image;
		if (input_image.rows != model->m_input_height|| input_image.cols != model->m_input_width) {
			cv::resize(input_image, resized_image, cv::Size(model->m_input_width, model->m_input_height));
		}
		else {
			resized_image = input_image.clone();
		}

		// 5. Mettre le modèle en mode évaluation
		model->eval();

		// 6. Convertir l'image en float
		cv::Mat img_float;
		resized_image.convertTo(img_float, CV_32F, 1.0 / 255.0);

		// 7. Créer un tenseur avec des vérifications
		auto options = torch::TensorOptions().dtype(torch::kFloat32);
		torch::Tensor img_tensor;
		try {
			img_tensor = torch::from_blob(img_float.data,
				{ resized_image.rows, resized_image.cols, 3 },
				options).clone();
		}
		catch (const std::exception& e) {
			std::string error = "Erreur lors de la conversion en tenseur: ";
			error += e.what();
			AfxMessageBox(CString(error.c_str()), MB_ICONERROR);
			return class_probabilities;
		}

		// 8. Permuter de [H, W, C] à [C, H, W] comme attendu par PyTorch
		img_tensor = img_tensor.permute({ 2, 0, 1 });

		// 9. Ajouter une dimension de batch si nécessaire
		if (img_tensor.dim() == 3) {
			img_tensor = img_tensor.unsqueeze(0);  // Ajouter une dimension de batch [1, C, H, W]
		}

		// 10. Déplacer le tensor sur le même périphérique que le modèle
		torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		torch::Device device(device_type);
		img_tensor = img_tensor.to(device);

		// 11. Désactiver le calcul du gradient
		torch::NoGradGuard no_grad;

		// 12. Faire l'inférence avec vérification supplémentaire
		torch::Tensor output;
		try {
			output = model->forward(img_tensor);
		}
		catch (const std::exception& e) {
			std::string error = "Erreur lors de l'inférence (format d'entrée incorrect): ";
			error += e.what();
			AfxMessageBox(CString(error.c_str()), MB_ICONERROR);
			return class_probabilities;
		}

		// 13. Obtenir les probabilités
		//auto probabilities = torch::exp(output);
		auto probabilities = torch::softmax(output, 1).squeeze(0); // [num_classes]


		// 14. Obtenir la classe prédite
		//auto max_result = torch::max(probabilities, 1);
		//auto predicted_idx = std::get<1>(max_result).item<int>();
		//auto confidence = std::get<0>(max_result).item<float>();

		//// 15. Obtenir le nom de la classe
		//std::string predicted_class;
		//if (class_names.empty()) {
		//	predicted_class = "Classe " + std::to_string(predicted_idx);
		//}
		//else if (predicted_idx < class_names.size()) {
		//	predicted_class = class_names[predicted_idx] + " (Conf: " +
		//		std::to_string(confidence * 100.0f).substr(0, 5) + "%)";
		//}
		//else {
		//	predicted_class = "Classe inconnue " + std::to_string(predicted_idx);
		//}
		for (size_t i = 0; i < class_names.size(); ++i) {
			if (i < probabilities.size(0)) {
				class_probabilities[class_names[i]] = probabilities[i].item<float>();
			}
		}
		
	}
	catch (const std::exception& e) {
		std::string error_msg = "Erreur lors de l'inférence: ";
		error_msg += e.what();
		AfxMessageBox(CString(error_msg.c_str()), MB_ICONERROR);
	
	}
	catch (...) {
		std::string error_msg = "Erreur inconnue lors de l'inférence";
		AfxMessageBox(CString(error_msg.c_str()), MB_ICONERROR);

	}

	return class_probabilities;
}

////// Fonction d'inférence pour une seule image
////std::string CNetcv::testImageInference(std::shared_ptr<CNNModelDyn> model, Mat & resized_image)
////{
////	try {
////		// Vérifier que le modèle existe
////		if (!model) {
////			AfxMessageBox(CString("Erreur modele non charge"),MB_OK);
////			return "";
////		}
////
////		// Mettre le modèle en mode évaluation
////		model->eval();
////
////		cv::Mat img_float;
////		resized_image.convertTo(img_float, CV_32F, 1.0 / 255.0);
////
////		// Créer un tenseur directement à partir des données de l'image
////		auto options = torch::TensorOptions().dtype(torch::kFloat32);
////		torch::Tensor img_tensor = torch::from_blob(img_float.data, { resized_image.rows, resized_image.cols, 3 }, options).clone();
////
////		// Permuter de [H, W, C] à [C, H, W] comme attendu par PyTorch
////		img_tensor = img_tensor.permute({ 2, 0, 1 });
////
////		// Déplacer le tensor sur le même périphérique que le modèle
////		torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
////		torch::Device device(device_type);
////		img_tensor = img_tensor.to(device);
////	
////		// Désactiver le calcul du gradient
////		torch::NoGradGuard no_grad;
////
////		// Faire l'inférence
////		auto output = model->forward(img_tensor);
////		auto probabilities = torch::exp(output);
////
////		// Obtenir la classe prédite
////		auto max_result = torch::max(probabilities, 1);
////		auto predicted_idx = std::get<1>(max_result).item<int>();
////		auto confidence = std::get<0>(max_result).item<float>();
////
////		// Obtenir le nom de la classe (si les noms de classe sont fournis)
////		std::string predicted_class;
////		if (class_names.empty()) {
////			predicted_class = "Classe " + std::to_string(predicted_idx);
////			return predicted_class;
////		}
////
////		if (predicted_idx < class_names.size()) 
////		{
////			predicted_class = class_names[predicted_idx];
////		}
////		else 
////		{
////			predicted_class = "Classe " + std::to_string(predicted_idx);
////		}
////
////		return predicted_class;
////
////	}
////	catch (const std::exception& e) {
////		std::string error_msg = "Erreur lors de l'inférence: ";
////		error_msg += e.what();
////		AfxMessageBox(CString(error_msg.c_str()), MB_ICONERROR);
////	}
////}

bool CNetcv::saveConfig(const std::vector<ConvLayerParams>& layers, const std::vector<DenseLayerParams>& denseLayers, const string& filePath)
{
	std::ofstream configFile(filePath);
	if (!configFile.is_open())
	{
		return false;
	}
	configFile << m_input_height << " " << m_input_width << " " << m_nb_classes << std::endl;
	configFile << layers.size() << std::endl;
	for (const auto& layer : layers)
	{
		configFile << layer.in_channels << " "
			<< layer.out_channels << " "
			<< layer.kernel_size << " "
			<< layer.stride << " "
			<< layer.padding << " "
			<< layer.use_pool << " "
			<< layer.pool_size << "\n";
	}

	// Sauvegarder les couches denses
	configFile << denseLayers.size() << std::endl;
	for (const auto& layer : denseLayers)
	{
		configFile << layer.nb_neurons << " " << layer.activation_type << std::endl;
	}

	configFile.close();
	return true;
}

bool CNetcv::loadConfig(const string& filePath, std::vector<ConvLayerParams>& convLayers,  std::vector<DenseLayerParams>& denseLayers)
{
	std::ifstream configFile(filePath);
	int num_layers;
	if (!configFile.is_open())
	{
		return false;
	}
	int input_height, input_width, num_classes;
	configFile >> input_height >> input_width >> num_classes;

	m_input_height = input_height;
	m_input_width = input_width;
	if (num_classes!=0) m_nb_classes = num_classes;
	

	size_t convLayerCount;
	configFile >> convLayerCount;
	convLayers.resize(convLayerCount);
	for (size_t i = 0; i < convLayerCount; ++i)
	{
		configFile >> convLayers[i].in_channels >> convLayers[i].out_channels >> convLayers[i].kernel_size
			>> convLayers[i].stride >> convLayers[i].padding >> convLayers[i].use_pool >> convLayers[i].pool_size;
	}

	size_t denseLayerCount;
	configFile >> denseLayerCount;
	denseLayers.resize(denseLayerCount);
	for (size_t i = 0; i < denseLayerCount; ++i)
	{
		configFile >> denseLayers[i].nb_neurons >> denseLayers[i].activation_type;
	}


	configFile.close();
	
	m_conv_params = convLayers;
	m_dens_params = denseLayers;
	return true;
}