import torch
import torch.nn as nn

def fc_block(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.ReLU()
    )

class FcModel(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.fc_layers = []
        self._make_fc_layers()

    def forward(self, x):
        x = self.fc_layers(x)
        return x

    def _make_fc_layers(self):
        in_size = self.in_features
        out_size = self.out_features
        if out_size <= in_size:
            r = int(in_size-out_size)//(self.num_layers)
        else:
            r = -int(out_size-in_size)//(self.num_layers)
    
        in_size -= r
        self.fc_layers.append(fc_block(self.in_features, in_size))
        for i in range(self.num_layers-2):
            self.fc_layers.append(fc_block(in_size, in_size-r))
            in_size -= r
            
        self.fc_layers.append(nn.Sequential(nn.Linear(in_size, self.out_features)))
        self.fc_layers = nn.Sequential(*self.fc_layers)

if __name__ == '__main__':
    model = FcModel(in_features=10, out_features=2, num_layers=3)
    data = torch.zeros(1, 10)
    out = model(data)
    print(out.shape)