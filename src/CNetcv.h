#pragma once
#pragma execution_character_set("utf-8")
#include <iostream>
#include <numeric>  // Pour std::accumulate
#include <map>
#include <algorithm>
#include <random>
#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem> 


using namespace torch;
using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace cv::dnn;


class CustomImageDataset : public torch::data::Dataset<CustomImageDataset> {
private:
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    int input_height_;
    int input_width_;

public:
    CustomImageDataset(const std::vector<std::string>& image_paths, const std::vector<int>& labels, int input_height, int input_width)
        : image_paths_(image_paths), labels_(labels), input_height_(input_height), input_width_(input_width) {
    }

    torch::data::Example<> get(size_t index) override {
        // Charger l'image depuis le disque
        cv::Mat img = cv::imread(image_paths_[index]);
        cv::resize(img, img, cv::Size(input_width_, input_height_));

        //// Convertir l'image en tenseur
        //torch::Tensor img_tensor = torch::from_blob(img.data, { 1, img.rows, img.cols, 3 }, torch::kByte);
        //img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32).div(255.0);

        // Conversion en format [H, W, C]
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F, 1.0 / 255.0);

        // Créer un tenseur directement à partir des données de l'image
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor img_tensor = torch::from_blob(img_float.data, { input_height_, input_width_, 3 }, options).clone();

        // Permuter de [H, W, C] à [C, H, W] comme attendu par PyTorch
        img_tensor = img_tensor.permute({ 2, 0, 1 });

        // Récupérer le label
        int label = labels_[index];
        torch::Tensor label_tensor = torch::tensor(label, torch::kLong);

        return { img_tensor, label_tensor};
    }

    torch::optional<size_t> size() const override {
        return image_paths_.size();
    }
};


struct CNNModel : torch::nn::Module {
	CNNModel(int num_classes) {
		// Utiliser un pattern d'initialisation plus simple
		conv1 = torch::nn::Conv2d(3, 32, 3);
		conv1->options.stride(1).padding(1);

		conv2 = torch::nn::Conv2d(32, 64, 3);
		conv2->options.stride(1).padding(1);

		fc = torch::nn::Linear(64 * 7 * 7, num_classes);

		// Enregistrer manuellement
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("fc", fc);
	}

	torch::Tensor forward(torch::Tensor x) {
		// Première couche convolutive + pooling
		x = torch::relu(conv1->forward(x));
		x = torch::max_pool2d(x, 2);
		// Deuxième couche convolutive + pooling
		x = torch::relu(conv2->forward(x));
		x = torch::max_pool2d(x, 2);
		// Aplatir et couche fully connected
		x = x.view({ -1, 64 * 7 * 7 });
		x = fc->forward(x);
		return torch::log_softmax(x, 1);
	}

	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
	torch::nn::Linear fc{ nullptr };
};

struct ConvLayerParams {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    bool use_pool;
    int pool_size;
};

struct DenseLayerParams
{
    int nb_neurons;
    int activation_type;
};

struct CNNModelDyn : torch::nn::Module {
    CNNModelDyn(const std::vector<ConvLayerParams>& conv_params, const std::vector<DenseLayerParams>& dense_params, 
        int input_height, int input_width, int num_classes)
    {
        // Stocker le nombre de couches
        num_layers = conv_params.size();

        // Dimension actuelle de l'image
        int current_height = input_height;
        int current_width = input_width;
		m_input_height = input_height;
		m_input_width = input_width;    

        // Créer les couches de convolution dynamiquement
        for (size_t i = 0; i < num_layers; ++i) {
            const auto& params = conv_params[i];

            // Créer la couche convolutive
            auto conv = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(params.in_channels, params.out_channels, params.kernel_size)
                .stride(params.stride)
                .padding(params.padding)
            );

            // Enregistrer la couche
            conv_layers.push_back(conv);
            register_module("conv" + std::to_string(i + 1), conv);

            // Mettre à jour les dimensions
            // Formule: (W - K + 2P) / S + 1
            current_height = (current_height - params.kernel_size + 2 * params.padding) / params.stride + 1;
            current_width = (current_width - params.kernel_size + 2 * params.padding) / params.stride + 1;

            // Si pooling utilisé, ajuster les dimensions
            if (params.use_pool) {
                current_height /= params.pool_size;
                current_width /= params.pool_size;
            }

            // Stocker les paramètres de pooling
            use_pool.push_back(params.use_pool);
            pool_sizes.push_back(params.pool_size);
        }

        // Calculer la taille d'entrée pour la première couche fully connected
        int flattened_size = conv_params.back().out_channels * current_height * current_width;

        // Créer les couches fully connected dynamiquement
        int input_size = flattened_size;
        for (size_t i = 0; i < dense_params.size(); ++i) {
            const auto& params = dense_params[i];

            // Créer la couche fully connected
            auto fc = torch::nn::Linear(input_size, params.nb_neurons);
            dense_layers.push_back(fc);
            register_module("fc" + std::to_string(i + 1), fc);

            // Stocker le type d'activation
            activation_types.push_back(params.activation_type);

            input_size = params.nb_neurons;
        }

        // Créer la dernière couche fully connected
        auto fc_final = torch::nn::Linear(input_size, num_classes);
        dense_layers.push_back(fc_final);
        register_module("fc_final", fc_final);

        // Stocker les dimensions finales pour la méthode forward
        final_channels = conv_params.back().out_channels;
        final_height = current_height;
        final_width = current_width;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // Passer à travers toutes les couches convolutives
        for (size_t i = 0; i < num_layers; ++i) {
            x = torch::relu(conv_layers[i]->forward(x));

            // Appliquer pooling si nécessaire
            if (use_pool[i]) {
                x = torch::max_pool2d(x, pool_sizes[i]);
            }
        }

        // Aplatir le tenseur
        x = x.view({ -1, final_channels * final_height * final_width });

        // Passer à travers toutes les couches fully connected
        for (size_t i = 0; i < dense_layers.size() - 1; ++i) {
            x = dense_layers[i]->forward(x);
            if (activation_types[i] == 0) {
                x = torch::relu(x);
            }
            else if (activation_types[i] == 1) {
                x = torch::sigmoid(x);
            }
        }

        // Dernière couche fully connected
        x = dense_layers.back()->forward(x);

        return torch::log_softmax(x, 1);
    }

    // Membres
    std::vector<torch::nn::Conv2d> conv_layers;
    std::vector<bool> use_pool;
    std::vector<int> pool_sizes;
    std::vector<torch::nn::Linear> dense_layers;
    std::vector<int> activation_types;

    // Pour conserver les dimensions
    size_t num_layers;
    int final_channels;
    int final_height;
    int final_width;

	int m_input_height;
	int m_input_width;

};



class CNetcv
{
public:
	CNetcv();


    //torch::Tensor m_train_image_tensor;
    //torch::Tensor m_train_label_tensor;
    //torch::Tensor m_test_image_tensor;
    //torch::Tensor m_test_label_tensor;


    std::vector<std::string> m_train_image_paths;
    std::vector<int> m_train_labels;
    std::vector<std::string> m_test_image_paths;
    std::vector<int> m_test_labels;


    std::vector<string> class_names;
    std::shared_ptr<CNNModelDyn> m_model;
	std::vector<ConvLayerParams> m_conv_params;
    std::vector<DenseLayerParams> m_dens_params;
	int m_nb_classes;
    int m_input_height;
    int m_input_width;
	double m_Accuracy;
	double m_Loss;
    BOOL m_L2reg;
    BOOL m_AutomLR;
	
	//int loadDataset(const string& datasetPath, vector<Mat>& images, vector<int>& labels);
	//void loadDataset(const string& datasetPath,  std::vector<cv::Mat>& images,  std::vector<int>& labels);
  //  bool loadDataset(const string& datasetPath, std::vector<cv::Mat>& train_images, std::vector<int>& train_labels, std::vector<cv::Mat>& test_images, std::vector<int>& test_labels);
    bool loadDataset(const string& datasetPath);
	void testInference(const vector<Mat>& testImages);
	void trainModel(Tensor& image_tensor, Tensor& label_tensor, int NbClasses, int num_epochs);  
	bool saveModel(std::shared_ptr<CNNModel> model, const std::string& filePath);
	std::shared_ptr<CNNModel> loadModel(const std::string& filePath, int numClasses);

    void trainModelDyn(int num_epochs,double lr,int batch_size, int Periode, string nomModel);
    //std::shared_ptr<CNNModelDyn> createModel(const std::vector<ConvLayerParams>& conv_params, int input_height, int input_width);
    std::shared_ptr<CNNModelDyn> createModel(const std::vector<ConvLayerParams>& conv_params, const std::vector<DenseLayerParams>& dense_params, int input_height, int input_width, int num_classes);
    bool saveModel( std::shared_ptr<CNNModelDyn> model,  const std::string& filePath,  const std::string& namesFilePath);
    std::shared_ptr<CNNModelDyn> loadModel(const std::string& filePath);
    std::map<std::string, float>  testImageInference(std::shared_ptr<CNNModelDyn>& model, Mat& resized_image);
    bool saveConfig(const std::vector<ConvLayerParams>& layers, const std::vector<DenseLayerParams>& denseLayers, const string& filePath,bool python);
    bool loadConfig(const string& filePath, std::vector<ConvLayerParams>& layers, std::vector<DenseLayerParams>& denseLayers);

};

