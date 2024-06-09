import glob
import random
from PIL import Image
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class MyDatasets(Dataset):
    def __init__(self, path, real_folder_name = "", fake_folder_name = "", train = True, transform = None):
        self.path = path
        
        if train:
            self.data_generated = glob.glob(self.path + f'{fake_folder_name}/*.*')
            self.data_real = glob.glob(self.path + f'{real_folder_name}/*.*')
            if len(self.data_generated) < len(self.data_real):
                self.data_real = random.sample(self.data_real, len(self.data_generated))
            self.data = self.data_real + self.data_generated
            self.class_list = ["real"] * len(self.data_real) + ["generated"] * len(self.data_generated)
        else:
            self.data = glob.glob(self.path + '/test/*.jpg')
            class_list = pl.read_csv("./datasets/test/test_labels.csv", separator = ';')
            self.class_list = class_list.select(pl.col('label')).to_numpy().flatten()
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path)
        label = self.class_list[idx]
        if self.transform:
            img = self.transform(img)
            
        return img, label


class DataProcesser():
    def __init__(self, size = 224):
        self.trans = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.rev_trans = transforms.Compose([transforms.Normalize((-1, -1, -1), (2, 2, 2)),
                                            transforms.ToPILImage()])
    
    def show_tensor_image(self, img): 
        if len(img.shape) == 4:
            b, c, h, w = img.shape
            for idx in range(b):
                t_img = img[idx, :, : ,:]
                t_img = self.rev_trans(t_img.type(torch.float64))
                plt.subplot(1, b, idx + 1)
                plt.imshow(t_img)
            
        else:
            img = self.rev_trans(img)
            plt.imshow(img)
            
        plt.show()

    def get_datasets(self, dataset_path, real_folder_name = "", fake_folder_name = "", train = True):
        return MyDatasets(path = dataset_path, real_folder_name=real_folder_name, fake_folder_name=fake_folder_name, train = train, transform = self.trans)