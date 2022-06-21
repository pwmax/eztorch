import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model):
        self.model = model
    
    def train(self, 
        loss, 
        device, 
        lr, 
        epoch, 
        save_path, 
        train_dataloader, 
        val_dataloader=None, 
        loss_data_fn=None, 
        input_data_transform=None, 
        out_data_transform=None
    ):
        
        assert(loss in ['mse', 'mae', 'crossentropy', 'binarycrossentropy'])
        loss_dict = nn.ModuleDict([
            ['mse', nn.MSELoss()],
            ['mae', nn.L1Loss()],
            ['crossentropy', nn.CrossEntropyLoss()],
            ['binarycrossentropy', nn.BCELoss()]
        ])
        
        self.model.to(device)
        criterion = loss_dict[loss]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_list = []
        val_loss_list = []

        for epoch in range(epoch):
            if val_dataloader:
                val_loss_list.extend(self.eval_model(val_dataloader, device, criterion, 
                                        input_data_transform, out_data_transform))

            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                if input_data_transform:
                    x, y = input_data_transform(x, y)
                for param in self.model.parameters():
                    param.grad = None
                out = self.model(x)
                if out_data_transform:
                    out = out_data_transform(out)
                rloss = criterion(out, y)
                rloss.backward()
                optim.step()
                loss_list.append(rloss.item())
                if loss_data_fn:
                    loss_data_fn(loss_list, val_loss_list)
                print(f'epoch {epoch}  loss {rloss.item():.9f}')
            self.save_model(f'{save_path}{epoch}-model.pth')


    def eval_model(self, val_dataloader, device, loss,
                    input_data_transform=None, out_data_transform=None):

        criterion = loss
        self.model.to(device)
        val_loss_list = []
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)
            if input_data_transform:
                x, y = input_data_transform(x, y)
            with torch.no_grad():
                out = self.model(x)
            if out_data_transform:
                out = out_data_transform(out)
            loss = criterion(out, y)
            val_loss_list.append(loss.item())
        return val_loss_list

    def update(self, input, target, loss, optimizer):
        for param in self.model.parameters():
            param.grad = None
        out = self.model(input)
        rloss = loss(out, target)
        rloss.backward()
        optimizer.step()
        return rloss

    def save_model(self, path):
        state = self.model.state_dict()
        torch.save(state, path)
    
    def load_model(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state)
