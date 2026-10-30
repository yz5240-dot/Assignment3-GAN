import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib import get_model, train_gan


def run(epochs=20, batch_size=128, output_dir='outputs', resume=False):
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training device:', device)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == 'cuda')
    model = get_model('GAN').to(device)
    optimizer = {
        'G': torch.optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        'D': torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    }
    model = train_gan(model, loader, nn.BCELoss(), optimizer, device=device,
                      epochs=epochs, output_dir=output_dir, resume=resume)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.epochs, args.batch_size, args.output_dir, args.resume)
