# basic
import os
import random
import sys
from PIL import Image
import scipy.io as sio
import numpy as np

# torch
import torch
# import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
from torchvision.datasets import FGVCAircraft


class RandomSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size

    def __iter__(self):
        return iter(np.random.choice(len(self.data_source), self.subset_size, replace=False))

    def __len__(self):
        return self.subset_size


class MVTec(torch.utils.data.Dataset):
    def __init__(self, dataset_name, path, class_name, transform=None, mask_transform=None, seed=0, split='train'):
        self.transform = transform
        self.mask_transform = mask_transform
        self.data = []

        if dataset_name == 'mvtec-ad-loco-ad':
            path = os.path.join(path, "mvtec-loco-ad", class_name)
            mv_str = '/000.'
        elif dataset_name == 'mvtec-ad-ad':
            path = os.path.join(path, "mvtec-ad", class_name)
            mv_str = '_mask.'
        else:
            path = os.path.join(path, "MPDD", class_name)
            mv_str = '_mask.'

        # normall folders
        normal_dir = os.path.join(path, split, "good")

        # normal samples
        for img_file in os.listdir(normal_dir):
            image_dir = os.path.join(normal_dir, img_file)
            self.data.append((image_dir, None))

        if split == 'test':
            # anomaly folder
            test_dir = os.path.join(path, "test")
            test_anomaly_dirs = []
            for entry in os.listdir(test_dir):
                full_path = os.path.join(test_dir, entry)

                # check if the entry is a directory and not the non-anomaly one
                if os.path.isdir(full_path) and full_path != normal_dir:
                    test_anomaly_dirs.append(full_path)

            # anomaly samples
            for dir in test_anomaly_dirs:
                for img_file in os.listdir(dir):
                    image_dir = os.path.join(dir, img_file)
                    mask_dir = image_dir.replace("test", "ground_truth")
                    parts = mask_dir.rsplit('.', 1)
                    mask_dir = parts[0] + mv_str + parts[1]
                    self.data.append((image_dir, mask_dir))

            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        if mask_path:
            mask = Image.open(mask_path).convert('RGB')
            mask = self.mask_transform(mask)
            mask = 1.0 - torch.all(mask == 0, dim=0).float()
            label = 1
        else:
            C, W, H = image.shape
            mask = torch.zeros((H, W))
            label = 0

        return image, label, mask

def prepare_loader(image_size, path, dataset_name, class_name, batch_size, test_batch_size, num_workers, seed, shots):
    transform = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])
    transform_fmnist = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS),
                                           transforms.Grayscale(num_output_channels=3),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                           ])
    mask_transform = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS),
                                         transforms.ToTensor()
                                         ])
    if dataset_name == 'mvtec-ad-loco-ad' or dataset_name == 'mvtec-ad-ad' or dataset_name == 'mpdd':
        train_set = MVTec(dataset_name, path, class_name, transform=transform, mask_transform=mask_transform, seed=seed,
                          split='train')
        test_set = MVTec(dataset_name, path, class_name, transform=transform, mask_transform=mask_transform, seed=seed,
                         split='test')
    elif dataset_name == 'fgvc-aircraft':
        train_dataset = FGVCAircraft(root=path, split='train', annotation_level='variant', transform=transform,
                                     download=True)
        test_dataset = FGVCAircraft(root=path, split='test', annotation_level='variant', transform=transform,
                                    download=True)

        desired_labels = [91, 96, 59, 19, 37, 45, 90, 68, 74, 89]

        train_set = [(data, 0) for (data, target) in train_dataset if target == int(class_name)]
        test_set_0 = [(data, 0) for (data, target) in test_dataset if target == int(class_name)]
        test_set_1 = [(data, 1) for (data, target) in test_dataset if
                      target in desired_labels and target != int(class_name)]

        num_zeros = len(test_set_0)
        num_ones = len(test_set_1)
        spacing = num_ones // (num_zeros + 1)

        # final test set with equally spaced 0's
        test_set = []
        index = 0
        for i in range(num_zeros):
            test_set.append(test_set_0[i])
            test_set.extend(test_set_1[index:index + spacing])
            index += spacing
        test_set.extend(test_set_1[index:])
    else:
        sys.exit("This is not a valid dataset name")

    if shots > 0 and shots < len(train_set):
        indices = list(range(shots))
        indices_seeded = [x + seed for x in indices]
        train_subset = torch.utils.data.Subset(train_set, indices_seeded)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=min(shots, batch_size), shuffle=True,
                                                   drop_last=True, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                                   pin_memory=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, drop_last=False,
                                              num_workers=num_workers)

    return train_loader, test_loader