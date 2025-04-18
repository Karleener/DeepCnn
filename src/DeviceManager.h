#pragma once
#include "pch.h"
#include <delayimp.h>
#include <Windows.h>

// Gestionnaire d'exception pour le chargement diff�r�
FARPROC WINAPI DelayLoadHook(unsigned dliNotify, PDelayLoadInfo pdli) {
	if (dliNotify == dliFailLoadLib) {
		// En cas d'�chec de chargement, retourner NULL sans erreur
		return NULL;
	}
	return NULL;
}

// D�finir le hook de chargement diff�r�
const PfnDliHook __pfnDliNotifyHook2 = DelayLoadHook;

//class DeviceManager {
//public:
//	static torch::Device getOptimalDevice() {
//		try {
//			// Essayer de charger torch_cuda.dll
//			HMODULE hTorchCuda = LoadLibraryA("torch_cuda.dll");
//			if (hTorchCuda && torch::cuda::is_available()) {
//				FreeLibrary(hTorchCuda); // Lib�ration sans effet car la DLL reste charg�e
//				return torch::Device(torch::kCUDA);
//			}
//		}
//		catch (...) {
//			// Ignorer toutes les exceptions
//		}
//
//		// Utiliser CPU par d�faut
//		return torch::Device(torch::kCPU);
//	}
//
//	static void moveToOptimalDevice(torch::nn::Module& model) {
//		model.to(getOptimalDevice());
//	}
//
//	static std::string getDeviceInfo() {
//		torch::Device device = getOptimalDevice();
//		if (device.type() == torch::kCUDA) {
//			std::string info = "CUDA disponible, utilisation du GPU: ";
//			//int deviceIndex = device.index();
//			//auto prop = torch::cuda::get_device_properties(deviceIndex);
//			//info += prop->name;
//			return info;
//		}
//		else {
//			return "CUDA non disponible, utilisation du CPU";
//		}
//	}
//};


