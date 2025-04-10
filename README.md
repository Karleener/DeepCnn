# DeepCnn
C++ Libtorch  based Interface for creation of convolutional neural network, train and test image classification, without python

Alpha test of a Windows exe  

SOURCE CODE WILL BE UPLOADED SOON !

# Third party dll are available here  
compiled version of Opencv version 4.10  
[decid 3D 2025_7001_release/opencv_world4100.dll](https://github.com/Karleener/Decid3D/blob/4c91c6335469a6f224ab6f51e2e41a1fe3e66ba3/decid%203D%202025_7001_release/opencv_world4100.dll)  
Libtorch 2.6 for windows, with cuda 12.6  
https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.6.0%2Bcu126.zip

Copy all DLL files (from Lib subdirectory of libtorch) in the same directory as DeepCnn2025.exe  

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
 ![test](images/train.png)

