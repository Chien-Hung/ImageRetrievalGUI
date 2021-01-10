import os
import os.path as osp
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Cars196Dataset_all(Dataset):
    def __init__(self, root_dir, transform):
        self.img_list = []
        for folder in os.listdir(root_dir):

            for img_name in os.listdir(osp.join(root_dir, folder)):
                self.img_list.append(osp.join(root_dir, folder, img_name))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)

        return self.img_list[idx], img


class Cars196Dataset(Dataset):
    def __init__(self, root_dir, transform, is_validation=True):
        self.img_list = []
        img_classes = sorted([x for x in os.listdir(root_dir)])
        if not is_validation:
            img_classes = img_classes[:len(img_classes)//2]
        else:
            img_classes = img_classes[len(img_classes)//2:]

        img_list = {i:sorted([root_dir + '/' + key + '/' + x for x in os.listdir(root_dir + '/' + key)]) for i, key in enumerate(img_classes)}
        img_list = [[img_path for img_path in img_list[key]] for key in img_list.keys()]
        self.img_list = [x for y in img_list for x in y]
        
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)

        return self.img_list[idx], img

class Cub200Dataset_all(Dataset):
    def __init__(self, root_dir, transform):
        self.img_list = []
        for folder in os.listdir(root_dir):

            for img_name in os.listdir(osp.join(root_dir, folder)):
                self.img_list.append(osp.join(root_dir, folder, img_name))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)

        return self.img_list[idx], img


class Cub200Dataset(Dataset):
    def __init__(self, root_dir, transform, is_validation=True):
        self.img_list = []
        img_classes = sorted([x for x in os.listdir(root_dir)])
        if not is_validation:
            img_classes = img_classes[:len(img_classes)//2]
        else:
            img_classes = img_classes[len(img_classes)//2:]

        img_list = {i:sorted([root_dir + '/' + key + '/' + x for x in os.listdir(root_dir + '/' + key)]) for i, key in enumerate(img_classes)}
        img_list = [[img_path for img_path in img_list[key]] for key in img_list.keys()]
        self.img_list = [x for y in img_list for x in y]
        
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)

        return self.img_list[idx], img

