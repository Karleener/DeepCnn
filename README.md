# DeepCnn 2025
C++ Libtorch  based Interface for creation of convolutional neural network, train and test image classification, without code    
Generation of equivalent python code automatically done using Mistral AI (need codestral API key to be stored in mistral_api_key.txt)

# Third party dll are available here  
Download compiled version of Opencv version 4.10  (file opencv_world4100.dll)  
https://opencv.org/blog/opencv-4-10-0/  
run opencv installation, and copy opencv_world4100.dll in the directory of DeepCnn2025.exe

# Libtorch 2.6 for windows, with cuda 12.6  
https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.6.0%2Bcu126.zip

Copy all DLL files (from Lib subdirectory of libtorch) in the same directory as DeepCnn2025.exe  

#Other dependencies : curl json (nlohmann)  : installation is easy using vcpkg

# Install visual c++ 2022 redistribuable  
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

# Run DeepCnn2025.exe 
![run DeppCnn2025](images/first.png)

# Run configuration to create a network structure and save it (.configCNN file)  

![configuration](images/confcnn.png)


Train the network selecting configCNN file and data directory structure including train and test subdirectories as follows  
(see sampledata.zip provided)  

 directory\train\classe1\image1.jpg  
 directory\train\classe1\image2.jpg  
 ...  
 directory\train\classe2\image1.jpg  
 directory\train\classe2\image1.jpg 
 directory\test\classe1\image1.jpg  
 directory\test\classe1\image2.jpg  
 ...  
 directory\test\classe2\image1.jpg  
 directory\test\classe2\image1.jpg  

 ![train](images/train.png)

# Wait for training completion (press F5 to stop the process)

 ![Training](images/courbes.png)

# Save training result as .pt file

# Test image classification selecting one of the test image.
# or multiclass image segmentation  
 ![test](images/class_test.png)  
 
It is possible to classify one single image or to create/label patches from images in order to create a train/test directory structure
Choose the blob size  
Clic on Select image and record patches - select saving folder.  
Open the training image containing objects to be localized.  
Left clic on several patches of the background (class 0)  
![Background](images/class_0.png)  
Right clic to increment the class number, then left clic on the patches of the current class  
![Class 1](images/class_1.png)  

Repeat the process for several classes. Close the training image.  
Clic on "Create train/test directories" and select the folder where you stored the patches.  
Close the Classification test dialog box and train your CNN.  
Open Classification test, choose the blob size and the image to be segmented.   
Segmented image (_segmented.png) and list of detected patches (csv file) are automatically saved.
![Segmentation result](images/segm.png)  

