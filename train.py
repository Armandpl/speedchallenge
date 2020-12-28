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
from models.resnet2p1d import generate_model
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

    # model = Resnet18Rnn()
    # init wandb
    run = wandb.init(project="speedchallenge", job_type='train')

    # download dataset
    # artifact = run.use_artifact("dataset_split:latest")
    # artifact_dir = artifact.download()

    # Hyperparams
    hyperparameter_defaults = dict(
        sequence_length = 20,
        learning_rate = 0.0001,
        batch_size = 64,
        num_epochs = 2,
        skip_frames = 1,
        model_depth = 34,
        max_target = 30,
        grayscale = False
    )

    # Pass your defaults to wandb.init
    run = wandb.init(config=hyperparameter_defaults)
    # run = wandb.init(project="speedchallenge")
    config = wandb.config

    # Init network
    # model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    if config.grayscale:
        model = generate_model(model_depth=config.model_depth, n_classes=1, n_input_channels=1)

        tfms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

        ])
    else:
        model = generate_model(model_depth=config.model_depth, n_classes=1, n_input_channels=3)
        tfms = None

    trainset = VideoFrameDataset(os.path.join("data", "train"), int(config.sequence_length),
        1, skip_frames=int(config.skip_frames), transform=tfms)

    validset = VideoFrameDataset(os.path.join("data", "valid"),
        int(config.sequence_length), 1, skip_frames=int(config.skip_frames), transform=tfms)

    print(len(trainset), " items in the training set")
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
            print(targets)
            targets = targets/config.max_target
            print(targets)
            targets = targets.to(device=device)
            print(targets)
            targets = targets.float().unsqueeze(1)
            print(targets)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            train_loss.update(float(loss.item())*config.max_target)

            # backward
            loss.backward()

            # gradient descent or adam step
            if (batch_idx+1)%config.batch_size == 0:
                print("batch: ", int(batch_idx/config.batch_size), "/", int(len(train_loader)/config.batch_size))
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                "train loss": train_loss.avg
                })

            if (batch_idx+1)%(len(train_loader)//8) == 0:
                evaluate(test_loader, model)

        train_loss.reset()


    # save model to wandb
    torch.save(model.state_dict(), 'model.pth')
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('model.pth')
    run.log_artifact(artifact)

