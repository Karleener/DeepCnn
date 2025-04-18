// pch.h : Il s'agit d'un fichier d'en-tête précompilé.
// Les fichiers listés ci-dessous sont compilés une seule fois, ce qui améliore les performances de génération des futures builds.
// Cela affecte également les performances d'IntelliSense, notamment la complétion du code et de nombreuses fonctionnalités de navigation du code.
// Toutefois, les fichiers listés ici sont TOUS recompilés si l'un d'entre eux est mis à jour entre les builds.
// N'ajoutez pas de fichiers fréquemment mis à jour ici, car cela annule les gains de performance.

#ifndef PCH_H
#define PCH_H
//#define NOMINMAX
// ajouter les en-têtes à précompiler ici

#include "framework.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <direct.h>
#include <io.h>
#include <torch/torch.h>
#include <c10/core/DeviceType.h>
#include <random>
#include <filesystem> 
#include <numeric>  // Pour std::accumulate
#include <map>
#include <algorithm>
#include <fstream>

#endif //PCH_H
