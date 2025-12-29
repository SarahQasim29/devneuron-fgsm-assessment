import torch
from model import MNISTNet

def load_pretrained_model(path="mnist_cnn.pth"):
    model = MNISTNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
