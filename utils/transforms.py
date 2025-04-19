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


def custom_transform(output_size):
    long_side = int(output_size * 1.25)

    transform_list = [
        transforms.Resize((output_size, long_side)),
        transforms.CenterCrop((output_size, output_size)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        RandomFIXEDRotation(),

        transforms.ColorJitter(brightness=(0.8, 1.1)),
        transforms.ColorJitter(contrast=(0.8, 1.1)),
        transforms.ColorJitter(saturation=(0.8, 1.1)),

        transforms.ToTensor(),
        ColorConstancyTransform(power=6, gamma=1.2),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ]

    return transforms.Compose(transform_list)


def simple_transform(output_size):
    return transforms.Compose([
        transforms.Resize((output_size, int(output_size * 1.25))),
        transforms.CenterCrop((output_size, output_size)),
        RandomFIXEDRotation(),
        transforms.ToTensor()
    ])
