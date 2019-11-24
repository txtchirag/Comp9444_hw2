import torch

torch.save(torch.load("model.pth",map_location=torch.device("cpu")), "./model.pth")