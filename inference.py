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


PATH = './HW1_net_best.pth'
with open('testing_img_order.txt') as f:
    test_images_list = [x.strip() for x in f.readlines()]

with open('classes.txt') as f:
    classes = [x.strip() for x in f.readlines()]

file_path = "testing_images/"


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


# Transform

transform_test = transforms.Compose([
    transforms.Lambda(pad),
    transforms.CenterCrop((375, 375)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset


class MyDataset(Dataset):
    def __init__(self, img_dir, img_list, transform):
        super(MyDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir + self.img_list[idx])
        img_name = self.img_list[idx]
        return (self.transform(img), img_name)


if __name__ == '__main__':
    submission = []
    net = torch.load(PATH)
    test_loader = DataLoader(MyDataset(file_path,
                                       test_images_list,
                                       transform_test),
                             batch_size=1,
                             shuffle=False,
                             num_workers=0)

    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_loader, 0):
            img, img_name = data
            outputs = net(img)
            _, predicted = torch.max(outputs, 1)
            submission.append([test_images_list[i], classes[predicted]])
            print(classes[predicted], predicted)
    np.savetxt('answer.txt', submission, fmt='%s')
