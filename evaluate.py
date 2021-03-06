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

# Check accuracy on training & test to see how good our model
def evaluate(loader, model):
    print("Evaluate")
    # Set model to eval
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y=y/config.max_target
            y = y.to(device=device)

            scores = model(x)
            loss = criterion(scores, y.float().unsqueeze(1))
            test_loss.update(float(loss.item())*config.max_target)

        wandb.log({
            "valid loss": test_loss.avg
        })
        test_loss.reset()

    # Set model back to train
    model.train()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda")

    # Hyperparams
    config = dict(
        sequence_length = 10,
        learning_rate = 0.0001,
        batch_size = 64,
        num_epochs = 3,
        skip_frames = 1,
        model_depth = 18,
        max_target = 30,
        grayscale = True
    )

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

    train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=validset, batch_size=1, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loss = AverageMeter()
    test_loss = AverageMeter()

    model.to(device)
    model.train()
    # Train Network
    for epoch in range(config.num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda 
            data = data.to(device=device)

            targets = targets/config.max_target
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets.float().unsqueeze(1))
            train_loss.update(float(loss.item())*config.max_target)

            # backward
            loss.backward()

            # gradient descent or adam step
            if (batch_idx+1)%config.batch_size == 0:
                print("batch: ", int(batch_idx/64), "/", int(len(train_loader)/64))
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                "train loss": train_loss.avg
                })

        train_loss.reset()

        evaluate(test_loader, model)

    # save model to wandb
    torch.save(model.state_dict(), 'model.pth')
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('model.pth')
    run.log_artifact(artifact)

