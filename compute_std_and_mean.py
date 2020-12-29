import torchvision
import torch

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_set = torchvision.datasets.ImageFolder('data/route', transform)
    loader = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=1)

    nimages = 0
    mean = 0.
    std = 0.
    for batch, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    # Final step
    mean /= nimages
    std /= nimages

    print(mean)
    print(std)
