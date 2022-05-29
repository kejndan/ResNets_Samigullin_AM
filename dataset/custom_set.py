import torch
import os
from configs.cfg_custom_set import cfg_custom_set
from torchvision import transforms
from torchvision.datasets import ImageFolder
from random import randint
import numpy as np


class CustomSet(torch.utils.data.Dataset):
    def __init__(self, ds_path, type_dataset, mixup_train=False, mixup_alpha=0.2):
        self.ds_path = ds_path
        self.type_dataset = type_dataset
        self.mixup_train = mixup_train
        self.mixup_alpha = mixup_alpha


        if type_dataset == 'train':
            self.transforms_for_img = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=[0.6, 1.4],
                                       saturation=[0.6, 1.4],
                                       hue=[0, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        elif type_dataset == 'valid' or type_dataset == 'test':
            self.transforms_for_img = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        self.data = ImageFolder(os.path.join(ds_path, type_dataset), self.transforms_for_img)


    def __getitem__(self, item):
        img_1, label_1 = self.data[item]

        labels_1 = torch.zeros(len(self.data.classes))
        labels_1[label_1] = 1.0

        if self.mixup_train:
            mixup_idx = randint(0, len(self)-1)
            img_2, label_2 = self.data[mixup_idx]

            labels_2 = torch.zeros(len(self.data.classes))
            labels_2[label_2] = 1.0


            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            img = lam * img_1 + (1 - lam) * img_2
            label = lam * labels_1 + (1 - lam) * labels_2
        else:
            img = img_1
            label = labels_1.argmax()

        return img, label

    def get_item_with_path(self, item):
        img, label = self.data[item]
        path = self.data.imgs[item][0]
        return img, label, path

    def __len__(self):
        return len(self.data.samples)

    def get_nb_elem_in_each_class(self):
        nb_imgs = [0] * len(self.data.classes)
        for img in self.data.imgs:
            nb_imgs[img[1]] += 1
        return nb_imgs





if __name__ == '__main__':
    ds = CustomSet(cfg_custom_set.path, 'train', mixup_train=True)
    img, lbl = ds[0]
    import matplotlib.pyplot as plt
    plt.imshow(img.permute(1,2,0).numpy())
    plt.show()
    print(img.sum(), lbl)
