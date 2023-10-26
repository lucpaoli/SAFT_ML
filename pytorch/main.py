from saftvrmie import SAFTVRMieNNParams, SAFTVRMieNN

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import MLP

# Define basic MLP into SAFT
class MLPSAFT(nn.Module):
    def __init__(self, fp_len, hidden_dim, saft_dim=5):
        super(MLPSAFT, self).__init__()
        self.fp_len = fp_len

        self.mlp = nn.Sequential(
            MLP(in_channels=fp_len, hidden_channels=hidden_dim),
            MLP(in_channels=hidden_dim, hidden_channels=saft_dim),
        )
        self.saftvrmie = SAFTVRMieNN()

    def forward(self, X):
        # Split X into fp, Mw, V, T
        #? Check if dim spec is right
        fp, Mw, V, T = torch.split(X, [self.fp_len, 1, 1, 1], dim=1)
        saft_params = self.mlp(fp)
        a_res = self.saftvrmie(saft_params, Mw, V, T)
        return a_res


if __name__ == "__main__":
    # Load in data
    # Create model
    model = MLPSAFT(128, 512)

    # Test forward pass
