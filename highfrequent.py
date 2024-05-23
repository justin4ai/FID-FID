import glob
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl

class MyDatasets(Dataset):
    def __init__(self, path, train = True, transform = None):
        self.path = path
        
        self.data = glob.glob(self.path + '/test/*.jpg')
        class_list = pl.read_csv("./datasets/test_labels.csv", separator = ';')
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

def show_tensor_image(t_img):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t*255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
    
    if len(t_img.shape) == 4:
        t_img = t_img[0, :, : ,:]
    img = reverse_transforms(t_img)
    plt.imshow(img)
    plt.show()

def get_datasets(size):
    trans = transforms.Compose([transforms.Resize((size, size)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda t: (t*2)-1)])

    return MyDatasets(path = "./datasets", transform = trans)


def main():
    img_size = 256
    data = get_datasets(img_size)
    dataloader = DataLoader(dataset = data, batch_size = 1)

    for batch in dataloader:
        img, label = batch
        print(img.shape, label)
        show_tensor_image(img)
        

        

if __name__ == '__main__':
    main()