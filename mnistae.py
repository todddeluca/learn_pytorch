# M
# modified from pytorch mnist classifier example
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# see also:
# https://blog.keras.io/building-autoencoders-in-keras.html
#

'''
mnist is 28x28x1 black and white channel (0 or 1).
'''

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class DenseAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(784, 128)
        self.dense2 = nn.Linear(128, 32)
        self.dense3 = nn.Linear(32, 128)
        self.dense4 = nn.Linear(128, 784)
        # self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # same padding
        # self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

        self.up1 = nn.Upsample

    def forward(self, x):
        """

        :param x: (batch, 28 * 28) image size as flattened batch
        :return:
        """
        # encoder
        x = self.dense1(x)  # b, 784 -> b, 128
        x = F.relu(x)
        x = self.dense2(x)  # b, 128 -> b, 32
        x = F.relu(x)

        # decoder
        x = self.dense3(x)  # b, 32 -> b, 128
        x = F.relu(x)
        x = self.dense4(x)  # b, 128 -> b, 784
        # x = F.sigmoid(x)

        return x  # b, 784, containing the logits of each pixel.

        # x = self.conv1(x)  # batch, 1, 28, 28 -> b, 32, 26, 26
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)  # -> b, 32, 13, 13
        # x = self.conv2(x)  # -> b, 64, 11, 11
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)  # -> b, 64, 5, 5
        # x = self.conv3(x)  # -> b, 128, 3, 3
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)  # -> b, 128, 1, 1
        #
        # # at this point the encoding is 128 dimensional
        #
        # # decoder
        # x = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
        #                          groups=1, bias=True, dilation=1, padding_mode='zeros')
        # x = torch.flatten(x, 1)
        # x = self.dropout1(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(data.size(0), -1)  # flatten batch (b, 1, 28, 28) -> (b, 784)
        # print(f'data.shape: {data.shape}')
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, data)

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = DenseAutoencoder().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
