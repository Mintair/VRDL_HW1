import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, Dataset

from PIL import Image

with open('training_labels.txt') as f:
    tokens = [x.strip().split() for x in f.readlines()]
# all the training images
train_images_list = [t[0] for t in tokens]
# all the training labels
train_labels = np.array([int(t[1].strip('0').split('.')[0]) - 1
                         for t in tokens])

file_path = "training_images/"

# Load dataset


class MyDataset(Dataset):
    def __init__(self, img_dir, img_list, labels, transform):
        super(MyDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = img_list
        self.transform = transform
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir + self.img_list[idx])
        label = self.labels[idx]
        return (self.transform(img), label)


def pad(img, size_max=500):
    """
    Pads images to the specified size (height x width).
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return transforms.functional.pad(
        img,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))


transform_train = transforms.Compose([
    transforms.Lambda(pad),
    transforms.CenterCrop((375, 375)),
    transforms.RandomOrder([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(15,)
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    train_loader = DataLoader(MyDataset(file_path,
                                        train_images_list,
                                        train_labels,
                                        transform_train),
                              batch_size=4,
                              shuffle=True,
                              num_workers=0)
    # Pretrained model: ResNet-50
    net_ft = models.resnet152(pretrained=True)
    num_ftrs = net_ft.fc.in_features
    net_ft.fc = nn.Linear(num_ftrs, 200)
    for child in net_ft.children():
        print(child)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(net_ft.parameters(), lr=0.001, momentum=0.9)
    cos_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,
                                                            T_max=10)
    net_ft = torch.nn.DataParallel(net_ft,
                                   device_ids=range(torch.cuda.device_count()))
    torch.backends.cudnn.benchmark = True
    net_ft = net_ft.cuda()

    for epoch in range(50):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            # forward + backward + optimize
            outputs = net_ft(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f'
                  % (epoch + 1, i + 1, running_loss / (i + 1)))
        cos_lr_scheduler.step()

    print('Finished Training')

    # Save trained neural network
    PATH = './HW1_net_cos.pth'
    torch.save(net_ft, PATH)
