import torch


torch.save(torch.load("_gpu.pth",map_location="cpu"), "./model_cpu.pth")