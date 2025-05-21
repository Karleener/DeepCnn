#pragma once
#include "CNetcv.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>

class CMistral
{
public:
	// ajouter une fonction de lecture de la clef API depuis un fichier "mistral_api_key.txt"
	CMistral(const std::string& apiKeyFilePath = "mistral_api_key.txt");
	// variables membres
	std::string apiKey;
	std::string generatePromptForConfig(const std::vector<ConvLayerParams>& convLayers,		const std::vector<DenseLayerParams>& denseLayers,int input_height, int input_width, int nb_classes);
	void  sendRequestToMistralAPI(const std::string& prompt, string PythonFile);
	static size_t  CMistral::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output);
	std::string generateTrainingPrompt(const std::vector<ConvLayerParams>& convLayers,
		const std::vector<DenseLayerParams>& denseLayers,
		int input_height, int input_width, int nb_classes, const std::string& datasetPath, int batch_size, int epochs, float learning_rate, bool from_scratch, const std::string& modelFilePath);


};

