from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim

import datetime


class image_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.define_model()
        self.data_path = "./data/"
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

    def load_cifar10_data(self):

        self.data = datasets.CIFAR10(
            self.data_path, train=True, download=True, transform=self.transform
        )
        self.data_val = datasets.CIFAR10(
            self.data_path, train=False, download=True, transform=self.transform
        )

    def define_model(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    # Same as the previous function, just a normal nn.sequential model
    def get_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Linear(8 * 8 * 8, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

        return model

    def forward(self, x_input):
        x = F.max_pool2d(torch.tanh(self.conv1(x_input)), 2)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), 2)
        x = x.view(-1, 8 * 8 * 8)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def training_loop(model, train_loader, loss_fn=None, n_epochs=100, optimizer=None):
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(
                "{} Epoch {}, Training loss {}".format(
                    datetime.datetime.now(), epoch, loss_train / len(train_loader)
                )
            )


def main():

    img_clf = image_classifier()

    img_clf.load_cifar10_data()

    label_map = {0: 0, 2: 1}
    cifar2 = [(img, label_map[label]) for img, label in img_clf.data if label in [0, 2]]

    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
    
    # cifar2_val = [
    #     (img, label_map[label]) for img, label in img_clf.data_val if label in [0, 2]
    # ]
    #val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=True)

    training_loop(img_clf, train_loader, n_epochs=10)


if __name__ == "__main__":
    main()
