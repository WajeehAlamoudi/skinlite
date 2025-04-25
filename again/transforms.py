import torch
import torchvision.transforms as transforms
import numpy as np
import random
import setting


class AddGaussianNoiseToRandomPixels:
    def __init__(self, noise_level=0.2, p=0.5):
        self.noise_level = noise_level
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            c, h, w = tensor.shape
            num_noisy_pixels = int(h * w * 0.1)
            for _ in range(num_noisy_pixels):
                i = random.randint(0, h - 1)
                j = random.randint(0, w - 1)
                tensor[:, i, j] += torch.randn(c) * self.noise_level
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor


class RandomFIXEDRotation:
    def __init__(self, angles=(0, 90, 180, 270), p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.rotate(img, random.choice(self.angles))
        return img


class ColorConstancyTransform:
    def __init__(self, power=6, gamma=1.0):
        self.power = power
        self.gamma = gamma

    def __call__(self, tensor):
        img = tensor.numpy()
        img = img ** (1 / self.gamma)
        norm = np.power(np.sum(np.power(img, self.power), axis=(1, 2), keepdims=True), 1 / self.power)
        img = img / norm
        img = np.clip(img, 0, 1)
        return torch.tensor(img, dtype=torch.float32)


train_transform = transforms.Compose([
    transforms.Resize((setting.IMAGE_SIZE, int(setting.IMAGE_SIZE*1.25))),
    transforms.CenterCrop((setting.IMAGE_SIZE, setting.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomFIXEDRotation(),
    transforms.ColorJitter(brightness=(0.8, 1.1)),
    transforms.ColorJitter(contrast=(0.8, 1.1)),
    transforms.ColorJitter(saturation=(0.8, 1.1)),
    transforms.ToTensor(),

])

val_transform = transforms.Compose([
    transforms.Resize((setting.IMAGE_SIZE, int(setting.IMAGE_SIZE*1.25))),
    transforms.CenterCrop((setting.IMAGE_SIZE, setting.IMAGE_SIZE)),
    transforms.ToTensor()
])
