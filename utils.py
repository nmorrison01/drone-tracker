import torch

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0") 
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    return device