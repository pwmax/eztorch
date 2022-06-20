import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model):
        self.model = model
    
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
