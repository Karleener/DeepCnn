#include "pch.h"
#include "CMistral.h"

using json = nlohmann::json;
// Callback function to handle the response data

// Fonction pour calculer la dimension du tensor après les couches convolutionnelles et de pooling
torch::Tensor calculate_tensor_dimensions(int input_height, int input_width, const std::vector<ConvLayerParams>& convLayers) {
	torch::Tensor input = torch::rand({ 1, 3, input_height, input_width }); // Exemple de tensor d'entrée

	for (const auto& layer : convLayers) {
		// Appliquez la convolution
		torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(layer.in_channels, layer.out_channels, layer.kernel_size)
			.stride(layer.stride)
			.padding(layer.padding));
		input = conv_layer->forward({ input });
		input = torch::relu(input);

		// Appliquez le pooling si nécessaire
		if (layer.use_pool) {
			torch::nn::MaxPool2d pool_layer(torch::nn::MaxPool2dOptions(layer.pool_size));
			input = pool_layer->forward({ input });
		}
	}

	return input;
}

size_t  CMistral::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output)
{
	size_t total_size = size * nmemb;
	output->append((char*)contents, total_size);
	return total_size;
}

std::string CMistral::generatePromptForConfig(const std::vector<ConvLayerParams>& convLayers,
	const std::vector<DenseLayerParams>& denseLayers,
	int input_height, int input_width, int nb_classes)
{
	std::string prompt = "Generate very precise Python code using PyTorch to train a CNN with the following steps:\n";
	prompt += "Calculate the tensor dimensions after each convolutional and pooling layer to ensure they match the expected size for the fully connected layer.\n";
	prompt += "syntaxe of comment has to be python compatible\n";
	prompt += "add import torch.nn.functional as F\n";
	prompt += "2. Create  the model with the following configuration:\n";
	prompt += "   - Input dimensions: " + std::to_string(input_height) + "x" + std::to_string(input_width) + "\n";
	prompt += "   - Number of classes: " + std::to_string(nb_classes) + "\n";
	prompt += "   - Convolutional Layers:\n";
	int current_height = input_height;
	int current_width = input_width;
	for (const auto& layer : convLayers)
	{
		prompt += "     - Channels: " + std::to_string(layer.in_channels) + " to " + std::to_string(layer.out_channels) + "\n";
		prompt += "       Kernel size: " + std::to_string(layer.kernel_size) + "\n";
		prompt += "       Stride: " + std::to_string(layer.stride) + "\n";
		prompt += "       Padding: " + std::to_string(layer.padding) + "\n";
		if (layer.use_pool) {
			prompt += "       Pooling: MaxPool with size " + std::to_string(layer.pool_size) + "\n";
			current_height /= layer.pool_size;
			current_width /= layer.pool_size;
		}

	}
	prompt += "Calculate dynamically the tensor dimensions for the dense layers from the conv layers and pooling and add following dense layers, creating a convs function un CNN class, based on random input\n";
	prompt += "   - Dense Layers:\n";
	for (size_t i = 0; i < denseLayers.size(); ++i) {
		prompt += "     - Neurons: " + std::to_string(denseLayers[i].nb_neurons) + "\n";
		if (denseLayers[i].activation_type == 0)
			prompt += "    Activation: Relu \n";
		else if (denseLayers[i].activation_type == 1)
			prompt += "    Activation: Sigmoid \n";
	}

	prompt += "Calculate the tensor dimensions for the output layer\n";
	prompt += "Output Layer:\n";
	prompt += "  - Neurons: " + std::to_string(nb_classes) + "\n";
	prompt += "    Activation: Softmax\n";

	std::ofstream promptfile("prompt config.txt");
	if (promptfile.is_open()) {
		promptfile << prompt;
		promptfile.close();
	}
	else {
		std::cerr << "Impossible d'ouvrir le fichier pour écrire." << std::endl;
	}

	return prompt;
}


std::string CMistral::generateTrainingPrompt(const std::vector<ConvLayerParams>& convLayers,
	const std::vector<DenseLayerParams>& denseLayers,
	int input_height, int input_width, int nb_classes, const std::string& datasetPath,	int batch_size, int epochs, float learning_rate,bool from_scratch, const std::string& modelFilePath)
{
	std::string prompt = "Generate very precise Python code using PyTorch to train a CNN with the following steps:\n";
	prompt += "Calculate the tensor dimensions after each convolutional and pooling layer to ensure they match the expected size for the fully connected layer.\n";
	prompt += "syntaxe of comment has to be python compatible\n";
	prompt += "define the os environnement variable KMP_DUPLICATE_LIB_OK as true \n";
	prompt += "add import torch.nn.functional as F\n";
	prompt += "use cuda if avalaible\n";
	prompt += "1. Load the dataset from the directory: " + datasetPath + " Directory structure should be	\ntrain\n  classe1\n   image1.jpg\n   image2.jpg\n  classe2\n   image1.jpg\n...\n\ntest\n  classe1\n   image1.jpg\n   image2.jpg\netc.\n";
	prompt += "2. Create  the model with the following configuration:\n";
	prompt += "   - Input dimensions: " + std::to_string(input_height) + "x" + std::to_string(input_width) + "\n";
	prompt += "   - Number of classes: " + std::to_string(nb_classes) + "\n";
	prompt += "   - Convolutional Layers:\n";
	int current_height = input_height;
	int current_width = input_width;
	for (const auto& layer : convLayers) 
	{
		prompt += "     - Channels: " + std::to_string(layer.in_channels) + " to " + std::to_string(layer.out_channels) + "\n";
		prompt += "       Kernel size: " + std::to_string(layer.kernel_size) + "\n";
		prompt += "       Stride: " + std::to_string(layer.stride) + "\n";
		prompt += "       Padding: " + std::to_string(layer.padding) + "\n";
		if (layer.use_pool) {
			prompt += "       Pooling: MaxPool with size " + std::to_string(layer.pool_size) + "\n";
			current_height /= layer.pool_size;
			current_width /= layer.pool_size;
		}

	}

	prompt += "Calculate dynamically the tensor dimensions for the dense layers from the conv layers and pooling and add following dense layers, creating a convs function un CNN class, based on random input\n";
	prompt += "   - Dense Layers:\n";
	for (size_t i = 0; i < denseLayers.size(); ++i) {
		//if (i == 0) {
		//	prompt += "     - Input dimension: " + std::to_string(flattened_size) + "\n";
		//}
		prompt += "     - Neurons: " + std::to_string(denseLayers[i].nb_neurons) + "\n";
		if (denseLayers[i].activation_type == 0)
			prompt += "    Activation: Relu \n";
		else if (denseLayers[i].activation_type == 1)
			prompt += "    Activation: Sigmoid \n";
	}




	prompt += "Calculate the tensor dimensions for the output layer\n";
	prompt += "Output Layer:\n";
	prompt += "  - Neurons: " + std::to_string(nb_classes) + "\n";
	prompt += "    Activation: Softmax\n";



	prompt += "4. Train the model with the following parameters:\n";
	prompt += "   - Batch size: " + std::to_string(batch_size) + "\n";
	prompt += "   - Epochs: " + std::to_string(epochs) + "\n";
	prompt += "   - Learning rate: " + std::to_string(learning_rate) + "\n";
	prompt += "   - From scratch: " + std::string(from_scratch ? "true" : "false") + "\n";
	prompt += "5. Save the trained model to: " + modelFilePath + "\n";
	prompt += "6. Display the training results, such as accuracy and loss.\n";
	prompt += "7. Plot the training and validation loss and accuracy curves.\n";


	std::ofstream promptfile("prompt training.txt");
	if (promptfile.is_open()) {
		promptfile << prompt;
		promptfile.close();
	}
	else {
		std::cerr << "Impossible d'ouvrir le fichier pour écrire." << std::endl;
	}

	return prompt;
}


void  CMistral::sendRequestToMistralAPI(const std::string& prompt, string PythonFile)
{
	CURL* curl;
	CURLcode res;
	std::string readBuffer;

	curl_global_init(CURL_GLOBAL_DEFAULT);
	curl = curl_easy_init();

	if (curl) {
		// Définir l'endpoint de l'API de Mistral
		curl_easy_setopt(curl, CURLOPT_URL, "https://codestral.mistral.ai/v1/chat/completions");

		// Définir les en-têtes nécessaires, y compris la clé API
		struct curl_slist* headers = NULL;
		headers = curl_slist_append(headers, "Content-Type: application/json");
		headers = curl_slist_append(headers, "Accept : application/json");
		string authorization = "Authorization: Bearer " + apiKey;
		headers = curl_slist_append(headers, authorization.c_str());
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		// Créer le corps de la requête JSON
		json jsonBody;
		jsonBody["model"] = "codestral-latest";
		jsonBody["messages"] = {
		  {
			  {"role", "user"},
			  {"content", prompt}
		  }
		};
		jsonBody["max_tokens"] = 2500;
		std::string jsonBodyStr = jsonBody.dump();

		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonBodyStr.c_str());

		// Définir la fonction de callback pour gérer les données de réponse
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

		// Exécuter la requête
		res = curl_easy_perform(curl);

		// Vérifier les erreurs
		if (res != CURLE_OK) {
			std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
		}
		else {
			//std::cout << "Response: " << readBuffer << std::endl;

			// Sauvegarder la réponse dans un fichier 
			std::ofstream outFile("response.txt");
			if (outFile.is_open()) {
				outFile << readBuffer;
				outFile.close();
			}
			else {
				std::cerr << "Impossible d'ouvrir le fichier pour écrire." << std::endl;
			}

			try {
				json responseJson = json::parse(readBuffer);
				std::string content = responseJson["choices"][0]["message"]["content"];

				// Sauvegarder le contenu dans un fichier .py
				// enlever les lignes avant le code python
				std::string::size_type pos = content.find("```python");
				if (pos != std::string::npos) {
					content = content.substr(pos + 9); // 9 pour la longueur de "```python"
				}
				else {
					std::cerr << "Aucun code Python trouvé dans la réponse." << std::endl;
					return;
				}
				// enlever les lignes après le code python
				pos = content.find("```");
				if (pos != std::string::npos) {
					content = content.substr(0, pos); // 3 pour la longueur de "```"
				}
				else {
					std::cerr << "Aucun code Python trouvé dans la réponse." << std::endl;
					return;
				}
				// ajouter un commentaire à la fin du code python avec date et heure de création
				std::time_t now = std::time(nullptr);
				std::tm* localTime = std::localtime(&now);
				char buffer[80];
				std::strftime(buffer, sizeof(buffer), "# Generated by DeepCnn and Mistral on %Y-%m-%d %H:%M:%S\n", localTime);
				content += buffer;

				// Sauvegarder le code Python dans un fichier
				std::ofstream outFile(PythonFile);
				if (outFile.is_open()) {
					outFile << content;
					outFile.close();
					//std::cout << "Code Python sauvegardé dans generated_code.py" << std::endl;
				}
				else {
					std::cerr << "Impossible d'ouvrir le fichier pour écrire." << std::endl;
				}
			}
			catch (const std::exception& e) {
				std::cerr << "Erreur lors de l'analyse JSON: " << e.what() << std::endl;
				MessageBox(NULL, L"Error during JSON parsing - see response.txt", L"Error", MB_OK | MB_ICONERROR);
			}

		}

		// Nettoyer
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
	}

	curl_global_cleanup();
}


CMistral::CMistral(const std::string& apiKeyFilePath)
{
	std::ifstream apiKeyFile(apiKeyFilePath);
	if (apiKeyFile.is_open()) {
		std::getline(apiKeyFile, apiKey);
		apiKeyFile.close();
	}
	else {
		std::cerr << "Unable to open API key file." << std::endl;
		MessageBox(NULL, L"Unable to open codestral API key file named mistral_api_key.txt.", L"Error", MB_OK | MB_ICONERROR);
	}
}