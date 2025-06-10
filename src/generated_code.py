Sure! Below is the Python code using PyTorch to create a Convolutional Neural Network (CNN) with the specified configuration:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dense Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)

        # Output Layer
        self.fc4 = nn.Linear(5, 10)

    def forward(self, x):
        # Convolutional Layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = x.view(-1, 128 * 3 * 3)

        # Dense Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        # Output Layer
        x = F.softmax(self.fc4(x), dim=1)

        return x

# Example usage
if __name__ == "__main__":
    model = CNN()
    print(model)

    # Create a random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)

    # Forward pass
    output = model(input_tensor)
    print(output)
```

### Explanation:
1. **Convolutional Layers**:
   - The first convolutional layer takes an input with 1 channel (grayscale image) and outputs 32 channels.
   - The second convolutional layer takes 32 channels as input and outputs 32 channels.
   - The third convolutional layer takes 32 channels as input and outputs 128 channels.
   - Each convolutional layer is followed by a max-pooling layer with a kernel size of 2.

2. **Dense Layers**:
   - The first dense layer has 10 neurons with ReLU activation.
   - The second dense layer has 10 neurons with ReLU activation.
   - The third dense layer has 5 neurons with Sigmoid activation.

3. **Output Layer**:
   - The output layer has 10 neurons with Softmax activation to classify the input into one of the 10 classes.

### Note:
- The input tensor shape is `(batch_size, channels, height, width)`, so for a single grayscale image of size 28x28, the shape is `(1, 1, 28, 28)`.
- The flattening step (`x.view(-1, 128 * 3 * 3)`) is necessary to convert the 3D tensor output from the convolutional layers into a 1D tensor for the fully connected layers. The dimensions `128 * 3 * 3` come from the final spatial dimensions after the convolutional and pooling layers.