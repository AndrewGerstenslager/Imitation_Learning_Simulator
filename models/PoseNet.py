import torch

class NN(torch.nn.Module):
    """
    Simple NN network. 
    Utilizing PyTorch
    """
    def __init__(self): 
        super().__init__()

        # Construct CNN
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*64**2, 3*64**2),
            torch.nn.BatchNorm1d(3*64**2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(3*64**2, 3*64**2),
            torch.nn.BatchNorm1d(3*64**2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(3*64**2, 4),
            torch.nn.Softmax()
        )

        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x): # Forward propogation
        result = self.layers(x)
        #out = torch.stack((result[:,0], self.sigmoid(result[:,1])), dim=1)
        return result
    