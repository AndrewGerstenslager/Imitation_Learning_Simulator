import torch

class CNN(torch.nn.Module):
    """
    Simple CNN network. 
    Utilizing PyTorch for MNIST dataset (28x28 features)
    """
    def __init__(self): 
        super().__init__()

        # Construct CNN
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, # greyscale image, 1 channel
                out_channels=16, # 16 kernels to use
                kernel_size=5, # kernel 5x5
                stride=1, # stride is 1
                padding=2 # padded by 2, no dimension reduction
            ),
            torch.nn.ReLU(), # Relu Activation
            torch.nn.MaxPool2d( # 32x32x16
                kernel_size=2, # 2x2 max pooling
            ),
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2
            ), # 16x16x32
            torch.nn.Flatten(1,-1), # Flatten output prior to Linear
            torch.nn.Linear(16*16*32, 2) # output turn angle and confidence
        )

        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x): # Forward propogation
        result = self.layers(x)
        out = torch.tensor([result[0], self.sigmoid(result[1])])
        return out
    