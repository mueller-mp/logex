import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as TF
import torchvision.datasets as dset

## CREDIT TO https://github.com/agaldran/balanced_mixup ##

# pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches
    

class Skin_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.CLASSES = ['nontumor_skin_dermis_dermis', 'nontumor_skin_epidermis_epidermis',
       'tumor_skin_naevus_naevus', 'tumor_skin_melanoma_melanoma',
       'nontumor_skin_subcutis_subcutis',
       'nontumor_skin_sebaceousglands_sebaceousglands',
       'tumor_skin_epithelial_bcc', 'tumor_skin_epithelial_sqcc',
       'nontumor_skin_muscle_skeletal',
       'nontumor_skin_chondraltissue_chondraltissue',
       'nontumor_skin_sweatglands_sweatglands',
       'nontumor_skin_necrosis_necrosis',
       'nontumor_skin_hairfollicle_hairfollicle',
       'nontumor_skin_nerves_nerves', 'nontumor_skin_vessel_vessel',
       'nontumor_skin_elastosis_elastosis']

        self.label_df = pd.read_csv(os.path.join(label_dir, f'skincancer-{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (395, 395), interpolation=cv2.INTER_AREA)

        x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).long()
    
class Skin_dataset_similar(Skin_dataset):
    def __init__(self, data_dir, label_dir, split,eval_trafo=False):
        self.data_dir = data_dir
        self.split = split

        self.CLASSES = ['nontumor_skin_dermis_dermis', 'nontumor_skin_epidermis_epidermis',
       'tumor_skin_naevus_naevus', 'tumor_skin_melanoma_melanoma',
       'nontumor_skin_subcutis_subcutis',
       'nontumor_skin_sebaceousglands_sebaceousglands',
       'tumor_skin_epithelial_bcc', 'tumor_skin_epithelial_sqcc',
       'nontumor_skin_muscle_skeletal',
       'nontumor_skin_chondraltissue_chondraltissue',
       'nontumor_skin_sweatglands_sweatglands',
       'nontumor_skin_necrosis_necrosis',
       'nontumor_skin_hairfollicle_hairfollicle',
       'nontumor_skin_nerves_nerves', 'nontumor_skin_vessel_vessel',
       'nontumor_skin_elastosis_elastosis']

        if split=='train':
            self.label_df = pd.read_csv(os.path.join(label_dir, f'skincancer-train-similar.csv'))
        else:
            self.label_df = pd.read_csv(os.path.join(label_dir, f'skincancer-{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train' and not eval_trafo:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

class Skin_dataset_PIL(Skin_dataset):
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        im_pil = transforms.ToTensor()(img)
        x=TF.resize(im_pil, (395), interpolation=Image.BICUBIC)
        x = self.transform(x)
        y = np.array(self.labels[idx])
        return x.float(), torch.from_numpy(y).long()
    
class Skin_dataset_similar_PIL(Skin_dataset_similar):
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        im_pil = transforms.ToTensor()(img)
        x=TF.resize(im_pil, (395), interpolation=Image.BICUBIC)
        x = self.transform(x)
        y = np.array(self.labels[idx])
        return x.float(), torch.from_numpy(y).long()


class Skin_dataset_similar_PIL_real_and_syn_targeted_maxconf_100(Skin_dataset_similar_PIL):
    def __init__(self, data_dir, label_dir,split='train'):

        self.split = split
        self.CLASSES = ['nontumor_skin_dermis_dermis', 'nontumor_skin_epidermis_epidermis',
       'tumor_skin_naevus_naevus', 'tumor_skin_melanoma_melanoma',
       'nontumor_skin_subcutis_subcutis',
       'nontumor_skin_sebaceousglands_sebaceousglands',
       'tumor_skin_epithelial_bcc', 'tumor_skin_epithelial_sqcc',
       'nontumor_skin_muscle_skeletal',
       'nontumor_skin_chondraltissue_chondraltissue',
       'nontumor_skin_sweatglands_sweatglands',
       'nontumor_skin_necrosis_necrosis',
       'nontumor_skin_hairfollicle_hairfollicle',
       'nontumor_skin_nerves_nerves', 'nontumor_skin_vessel_vessel',
       'nontumor_skin_elastosis_elastosis']

        # nat
        self.data_dir_nat = data_dir
        if split=='train':
            self.label_df_nat = pd.read_csv(os.path.join(label_dir, f'skincancer-train-similar.csv'))
        else:
            self.label_df_nat = pd.read_csv(os.path.join(label_dir, f'skincancer-{split}.csv'))
        self.img_paths_nat = self.label_df_nat['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()

        if split=='train':
            # add syn
            self.data_dir_syn = 'AEDG/syn_tail_images/ddim_3.0_20_conf_sgd_20_0.01_none_latents_0.1_cond_pt_uncond_pt_noise_const_0.005earlyS0.4/'
            self.label_df_syn = pd.read_csv(os.path.join(label_dir, f'similar_pil_targeted_maxconf_100_conf04.csv'))
            self.img_paths_syn = self.label_df_syn['id'].apply(lambda x: os.path.join(self.data_dir_syn, x)).values.tolist()

            self.label_df = pd.concat([self.label_df_nat, self.label_df_syn])
            self.img_paths = self.img_paths_nat+self.img_paths_syn

            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

        else:
            self.label_df = self.label_df_nat
            self.img_paths = self.img_paths_nat
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        self.label_df[self.CLASSES]=self.label_df[self.CLASSES].astype('int')
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values
        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

class Skin_dataset_similar_PIL_tail_syn_targeted_maxconf_100(Skin_dataset_similar_PIL):
    def __init__(self, data_dir, label_dir,split='train'):
        self.data_dir = 'AEDG/syn_tail_images/ddim_3.0_20_conf_sgd_20_0.01_none_latents_0.1_cond_pt_uncond_pt_noise_const_0.005earlyS0.4/'
        self.split=split
        self.CLASSES = ['nontumor_skin_dermis_dermis',
 'nontumor_skin_epidermis_epidermis',
 'tumor_skin_naevus_naevus',
 'nontumor_skin_subcutis_subcutis',
 'nontumor_skin_sebaceousglands_sebaceousglands',
 'tumor_skin_epithelial_bcc',
 'nontumor_skin_muscle_skeletal',
 'nontumor_skin_chondraltissue_chondraltissue',
 'nontumor_skin_sweatglands_sweatglands',
 'nontumor_skin_necrosis_necrosis',
 'nontumor_skin_vessel_vessel',
 'nontumor_skin_elastosis_elastosis']
        
        self.label_df = pd.read_csv(os.path.join(label_dir, f'similar_pil_targeted_maxconf_100_conf04.csv'))
        self.label_df[self.CLASSES]=self.label_df[self.CLASSES].astype('int')

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(self.data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
