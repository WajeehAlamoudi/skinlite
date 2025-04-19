import torch
from torchvision import transforms
import random
import torchvision.transforms.functional as F


class ColorConstancyTransform:
    def __init__(self, power=6, gamma=1.2):
        self.power = power
        self.gamma = gamma

    def __call__(self, img):
        if self.gamma is not None:
            img = img.clamp(min=1e-8)  # avoid zeros
            img = img.pow(self.gamma)

        img_power = img.pow(self.power)
        mean_power = img_power.mean(dim=(1, 2))
        norm = mean_power.pow(1.0 / self.power)

        img = img / norm[:, None, None]
        img = img / img.max()

        return img.clamp(0.0, 1.0)


class RandomFIXEDRotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return F.rotate(img, angle)


class AddGaussianNoiseToRandomPixels:
    def __init__(self, noise_prob=0.05, mean=0.0, std=0.1):
        """
        noise_prob: float [0, 1], fraction of pixels to apply noise to
        mean: mean of Gaussian noise
        std: standard deviation of Gaussian noise
        """
        self.noise_prob = noise_prob
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Create a noise mask with True for pixels to be noised
        noise_mask = torch.rand_like(tensor) < self.noise_prob
        # Generate noise
        noise = torch.randn_like(tensor) * self.std + self.mean
        # Apply noise only where mask is True
        return tensor + noise * noise_mask


def custom_transform(output_size):
    long_side = int(output_size * 1.25)

    transform_list = [
        # 1. resize + square crop
        transforms.Resize((output_size, long_side)),
        transforms.CenterCrop((output_size, output_size)),
        # 2. Geometric transformations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomFIXEDRotation(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # 3. Color/intensity transformations (applied to PIL images)
        transforms.ColorJitter(brightness=(0.8, 1.1)),
        transforms.ColorJitter(contrast=(0.8, 1.1)),
        transforms.ColorJitter(saturation=(0.8, 1.1)),

        # 5. Tensor-based operations
        transforms.ToTensor(),
        ColorConstancyTransform(power=6, gamma=1.2),
        AddGaussianNoiseToRandomPixels(),

        # 6. Normalization always comes last
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ]

    return transforms.Compose(transform_list)


def simple_transform(output_size):
    return transforms.Compose([
        # 1. resize + square crop
        transforms.Resize((output_size, int(output_size * 1.25))),
        transforms.CenterCrop((output_size, output_size)),
        # 2. Tensor-based operations
        transforms.ToTensor(),
        ColorConstancyTransform(power=6, gamma=1.2),
        # 3. Normalization always comes last
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
