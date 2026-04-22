import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # For M1/M2 Macs
