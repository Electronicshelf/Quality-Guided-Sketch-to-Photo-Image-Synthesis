from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
'''
#### cite
'``
author={Yunjey Choi and Minje Choi and Munyoung Kim and Jung-Woo Ha and Sunghun Kim and Jaegul Choo},
Adapted from >> title={StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},
}
```


'''

class CelebA(data.Dataset):
    """ Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
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


        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()

        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            # print(i)
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        print("finito")
        print('Finished preprocessing the CelebA dataset...')

    def get_dataX(self):
        dataset = self.train_dataset
        return dataset

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        # print(filename, label )
        # exit()
        image  = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        self.imageX = get_2nd_dir(self.image_dir)
        self.imageX = Image.open(os.path.join(self.imageX, filename)).convert('RGB')
        return self.transform(image), self.transform(self.imageX), torch.FloatTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_2nd_dir(dir):
    """Select the second directory ."""
    x = "data"
    A = "CelebA-HQ-White-Background"
    B = "CelebA-HQ-White-Background-Sketch"
    dir_domain = str(dir).split("/")
    # print(dir_domain)
    # exit()
    dir_domain = dir_domain[0]


    if dir_domain == A:
        print(os.path.join(x, B))
        return os.path.join(x, B)
    else:
        # print(os.path.join(x, A))
        return os.path.join(x, A)


def get_loader(image_dir, attr_path, selected_attrs, crop_size, image_size,
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
    # print(image_dir, attr_path)
    # exit()
    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
        dataX   = CelebA(image_dir, attr_path, selected_attrs, transform, mode).get_dataX()

    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader, dataX
