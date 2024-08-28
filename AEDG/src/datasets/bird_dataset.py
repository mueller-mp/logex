""" CUB-200-2011 (Bird) Dataset"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset

image_path = {}
image_label = {}


class BirdDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, path, phase='train', transform=None):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.image_id = []
        self.num_classes = 200
        self.path = path
        # get image path from images.txt
        with open(os.path.join(path, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(path, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(path, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        self.transform = transform

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(self.path, 'images', image_path[image_id]))
        if self.transform is not None:
            image = self.transform(image)

        # return image and label
        return image, image_label[image_id] - 1  # count begin from zero

    def __len__(self):
        return len(self.image_id)

