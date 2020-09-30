from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """ Data-set class for the CelebA data-set."""
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and pre-process the CelebA data-set."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.imageX = None
        self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Pre-process the CelebA attribute file."""
        filenames = os.listdir(self.image_dir)
        random.seed(1234)
        random.shuffle(filenames)
        for i, line in enumerate(filenames):
            self.test_dataset.append(line)
        print('Finished pre-processing the CelebA data set...')

    def get_dataX(self):
        """Return Dataset """
        dataset = self.test_dataset
        return dataset

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset     = self.test_dataset
        filename    = dataset[index]
        image       = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        self.imageX = get_2nd_dir()
        self.imageX = Image.open(os.path.join(self.imageX, filename)).convert('RGB')

        return self.transform(image), self.transform(self.imageX), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_2nd_dir():
    """Select the second directory."""
    x = "data"
    A = "test_Sketch"
    return os.path.join(x, A)

def get_loader_gui(image_dir, attr_path, selected_attrs, image_size,
                   batch_size=5, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    dataX = None
    transform = []
    # if mode == 'train':
        # transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)


    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader
