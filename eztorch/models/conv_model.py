import torch
import torch.nn as nn

def conv3x3_pool(in_channels, out_channels, max_pool=False):
    if max_pool:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
        )

class ConvModel(nn.Module):
    def __init__(self, model_size, num_layers, in_channels, out_classes):
        super().__init__()
        assert(2 <= num_layers <= 6)
        assert(out_classes <= 128)
        assert(model_size in ['small', 'tiny'])
        self.model_size = model_size
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.conv_layers = []
        self.fc_layers = []
        self._make_conv_layers()
        self._make_fc_layers()
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def load_state(self, state_path):
        state = torch.load(state_path)
        self.load_state_dict(state)
    
    def save_state(self, save_path):
        torch.save(self.state_dict(), save_path)

    def _make_conv_layers(self):
        l = [2, 2, 2, 3, 3, 3]
        c = [4, 4, 8, 8, 16, 16]
        if self.model_size == 'tiny':
            ms = 1
        if self.model_size == 'small':
            ms = 3
    
        self.conv_layers.append(conv3x3_pool(self.in_channels, c[0]*ms))
        for i in range(self.num_layers):
            if i != 0:
                self.conv_layers.append(conv3x3_pool(c[i-1]*ms, c[i]*ms))
            for t in range(l[i]-2):
                self.conv_layers.append(conv3x3_pool(c[i]*ms, c[i]*ms))
            self.conv_layers.append(conv3x3_pool(c[i]*ms, c[i]*ms, max_pool=True))
        
        if self.model_size == 'tiny':
            self.conv_layers.append(conv3x3_pool(c[self.num_layers-1]*ms, 16))

        if self.model_size == 'small':
            self.conv_layers.append(conv3x3_pool(c[self.num_layers-1]*ms, 48))

        self.conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv_layers = nn.Sequential(*self.conv_layers)
    
    def _make_fc_layers(self):
        if self.model_size == 'tiny':
            self.fc_layers = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, self.out_classes),
            )
        
        if self.model_size == 'small':
            self.fc_layers = nn.Sequential(
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Linear(128, self.out_classes),
            )

if __name__ == '__main__':
    model = ConvModel(model_size='tiny', num_layers=2, in_channels=3, out_classes=100)
    data = torch.zeros(1, 3, 256, 256)
    out = model(data)
    print(out.shape)
    model.info()