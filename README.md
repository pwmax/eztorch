# eztorch
Train pytorch models in ~~one line~~

### Example
```
from eztorch.trainer import Trainer

model = Model()
trainer = Trainer(model)
trainer.load_model('model.pth')

train_dataset = TrainDataset()
train_dataloader = DataLoader(train_dataset)
val_dataset = ValDataset()
val_dataloader = DataLoader(val_dataset)

def show_loss(train_loss, val_loss, epoch):
  print(f'epocdh {epoch} loss {train_loss[-1]:.9f}') 
  if keyboard.is_pressed('3'):
      plt.ylim(0, 10)
      plt.plot(train_loss, "-r", label="train loss")
      plt.plot(val_loss, "-b", label="val loss")
      plt.legend(loc="upper right")
      plt.show()

def input_transform(x, y):
  return (x.long(), y.long())

def out_transform(out):
  return out
    
trainer.train(
    loss='crossentropy', 
    device=device, 
    lr=lr, 
    epoch=num_epoch, 
    save_path=save_path, 
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_data_fn=show_loss,
    input_data_transform=input_transform,
    out_data_transform=out_transform
) 

```
### Installation
```
pip3 install git+https://github.com/pwmax/eztorch.git
```
