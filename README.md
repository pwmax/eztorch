# eztorch
Train pytorch models in one line

### Example
```
from eztorch.trainer import Trainer
from eztorch.conv_model import ConvModel
from eztorch.fc_model import FcModel

model = FcModel(in_features=784, out_features=10, num_layers=3)
trainer = Trainer(model)
dataset = TrainData()
dataloader = DataLoader()

trainer.classification(epoch=10, dataloader=dataloader, 
                        lr=1e-3, device='cpu', save_path='model.pth')

trainer.regression(epoch=10, dataloader=dataloader, 
                    lr=1e-3, device='cpu', save_path='model.pth')

trainer.load_model('model.pth')
for x, y in dataloader:
    x, y = x.float(), y.float()
    rloss = trainer.update(x, y, loss, optim)
    print('loss = {rloss.item()}')
trainer.save_model('model.pth')
```
### Installation
```
pip3 install git+https://github.com/pwmax/eztorch.git
```
