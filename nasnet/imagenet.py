import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


# inception preprocessing

class ImageNet(datasets.ImageFolder):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(self, root, train=True, image_size=224, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        traindir = os.path.join(self.root, 'train')
        valdir = os.path.join(self.root, 'val')
        self.train = train
        self.image_size = image_size
        transform = transform or self.preprocess()
        super(ImageNet, self).__init__(train and traindir or valdir,
                                       transform=transform, target_transform=target_transform)

    def preprocess(self):
        if self.train:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((int(self.image_size / 0.875), int(self.image_size / 0.875))),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
