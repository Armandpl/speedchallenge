# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

##
from video_dataset import VideoFrameDataset, ImglistToTensor
from utils import AverageMeter
from models.resnet import generate_model
import wandb
import os
from progressbar import progressbar
from models.cnn_lstm import Resnet18Rnn

if __name__ == "__main__":
    run = wandb.init()
    artifact = run.use_artifact('armandpl/speedchallenge/model:v18', type='model')
    artifact_dir = artifact.download()

    # Set device
    device = torch.device("cuda")

    config = wandb.config
    config.sequence_length = 10
    config.skip_frames = 1
    config.model_depth = 18
    config.max_target = 30
    config.grayscale = True

    if config.grayscale:
        model = generate_model(model_depth=config.model_depth, n_classes=1, n_input_channels=1)

        tfms = transforms.Compose([
            transforms.Grayscale()
        ])
    else: 
        model = generate_model(model_depth=config.model_depth, n_classes=1, n_input_channels=3)
        tfms = None

    validset = VideoFrameDataset(os.path.join("data", "route"), 
        int(config.sequence_length), 1, skip_frames=int(config.skip_frames), transform=tfms)

    print(len(validset), " items in the validation set")

    test_loader = DataLoader(dataset=validset, batch_size=1, shuffle=False)

    checkpoint = torch.load(os.path.join(artifact_dir,'model.pth'))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    res = open('test.txt', 'w')
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device=device)
            y=y/config.max_target
            y = y.to(device=device)

            scores = model(x)
            speed = scores.cpu().item()*config.max_target
            print(speed)
            res.write(str(round(speed, 1))+'\n')
