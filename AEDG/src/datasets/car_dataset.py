""" Stanford Cars (Car) Dataset """
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

DATAPATH = './stanford_cars/'


class CarDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Cars images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.num_classes = 196

        self.images = []
        self.labels = []

        meta_mat = 'devkit/cars_meta.mat'
        meta_mat = loadmat(os.path.join(DATAPATH, meta_mat))

        self.class_names = [class_name.item() for class_name in meta_mat['class_names'][0]]
        if phase == 'train':
            mat_file = 'devkit/cars_train_annos.mat'
            subdir = 'cars_train'
        else:
            mat_file = 'devkit/cars_test_annos_withlabels.mat'
            subdir = 'cars_test'

        list_path = os.path.join(DATAPATH, mat_file)

        list_mat = loadmat(list_path)
        num_inst = len(list_mat['annotations']['fname'][0])
        for i in range(num_inst):
            path = list_mat['annotations']['fname'][0][i].item()
            path = os.path.join(subdir, path)
            label = list_mat['annotations']['class'][0][i].item()
            self.images.append(path)
            self.labels.append(label)

        print('Car Dataset with {} instances for {} phase'.format(len(self.images), self.phase))

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item] - 1  # count begin from zero

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = CarDataset('val', resize=[500,500])
    # print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        # print(image.shape, label)
