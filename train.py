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
        for batch_idx, (x, y) in progressbar(enumerate(loader)):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            loss = criterion(scores, y.float().unsqueeze(1))
            test_loss.update(float(loss.item()))

        wandb.log({
            "valid loss": test_loss.avg
        })
        test_loss.reset()

    # Set model back to train
    model.train()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda")

    model = generate_model(model_depth=34, n_classes=1)
    # model = Resnet18Rnn()
    # init wandb
    run = wandb.init(project="speedchallenge", job_type='train')

    # download dataset
    # artifact = run.use_artifact("dataset_split:latest")
    # artifact_dir = artifact.download()

    # Hyperparams
    hyperparameter_defaults = dict(
        sequence_length = 10,
        learning_rate = 0.0001,
        batch_size = 32,
        num_epochs = 20,
        skip_frames = 8,
        model = "3D Resnet34"
        )

    # Pass your defaults to wandb.init
    run = wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    trainset = VideoFrameDataset(os.path.join("data", "train"), config.sequence_length, config.sequence_length*config.skip_frames/2, skip_frames=config.skip_frames)
    validset = VideoFrameDataset(os.path.join("data", "valid"), config.sequence_length, 10, skip_frames=config.skip_frames)

    train_loader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=validset, batch_size=config.batch_size, shuffle=True)

    # Initialize network
    # model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)


    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loss = AverageMeter()
    test_loss = AverageMeter()

    model.to(device)
    model.train()
    # Train Network
    for epoch in range(config.num_epochs):
        for batch_idx, (data, targets) in progressbar(enumerate(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets.float().unsqueeze(1))
            wandb.log({
                "train loss": float(loss.item())
            })
            train_loss.update(float(loss.item()))

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        wandb.log({
        "train loss": train_loss.avg
        })
        train_loss.reset()

        evaluate(test_loader, model)

