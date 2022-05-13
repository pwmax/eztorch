import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model):
        self.model = model
        self.loss = nn.ModuleDict([
            ['mse', nn.MSELoss()],
            ['mae', nn.L1Loss()],
            ['cel', nn.CrossEntropyLoss()],
        ])

    def classification(self, epoch, dataloader, lr, device, save_path, model_state=None):
        model = self.model
        if model_state:
            model.load_state(model_state)
        model.train()
        model.to(device)

        criterion = self.loss['cel']
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        
        for e in range(epoch):
            for x, y in dataloader:
                x, y = self._data_transform(x, y, device, 'c')
                optim.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optim.step()
        
                print('epoch(%i/%i) loss %.7f' % (e, epoch, loss.item()))
        model.save_state(save_path)
    
    def regression(self, epoch, dataloader, lr, device, save_path, loss='mse', model_state=None):
        model = self.model
        if model_state:
            model.load_state(model_state)
        model.train()
        model.to(device)

        criterion = self.loss[loss]
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        
        for e in range(epoch):
            for x, y in dataloader:
                x, y = self._data_transform(x, y, device, 'r')
                optim.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optim.step()
        
                print('epoch(%i/%i) loss %.7f' % (e, epoch, loss.item()))
        model.save_state(save_path)

    def _data_transform(self, x, y, device, task):
        in_shape = self._model_input_shape()
        x = x.to(device)
        y = y.to(device)

        if len(in_shape) == 2:
            x = x.reshape(x.size(0), -1)

        if len(in_shape) == 4:
            if len(x.shape) == 3:
                x = x.reshape(x.size(0), 1, x.size(1), x.size(2))

        if task == 'r':
            x, y = x.float(), y.float()
        
        if task == 'c':
            x, y = x.float(), y.long()
            y = y.reshape(-1)
            
        return x, y

    def _model_input_shape(self):
        p = next(iter(self.model.parameters()))
        p = p.shape
        return p