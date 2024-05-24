import glob
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import polars as pl
import math

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

def HighPass(img):
    img_num = img.shape[0]
    high_frequent_img = torch.empty_like(img)
    for i in range(img_num):
        original_img = img[i, :, :, :]
        c, h, w = original_img.shape
        fft_image = torch.fft.fftshift(torch.fft.fft2(original_img, dim = [-2, -1], norm = 'ortho'))
        fft_image[:, h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8] = 0.0
        high_passed_image = torch.fft.ifft2(torch.fft.ifftshift(fft_image), dim = [-2, -1], norm = 'ortho')
        high_frequent_img[i, :, :, :] = high_passed_image
        
    return high_frequent_img

def PatchEmbadding(img, patch_num):
    b, c, h, w = img.shape
    patch_size = h//patch_num
    embaddings_size = c * patch_size ** 2
    
    projection = nn.Conv2d(c, embaddings_size, kernel_size = patch_size, stride = patch_size)
    patch_embaddings = projection(img).flatten(2).transpose(1, 2)
    
    position_embaddings = nn.Parameter(torch.randn(1, patch_num**2, embaddings_size))
    patch_embaddings = patch_embaddings + position_embaddings
    
    return patch_embaddings

def to_score(patch_embaddings):
    b, patch_num, embadding_size = patch_embaddings.shape
    num_heads = 12
    head_size = int(embadding_size/num_heads)
    new_shape = patch_embaddings.size()[:-1] + (num_heads, head_size)
    patch_embaddings = patch_embaddings.view(new_shape)
    patch_embaddings = patch_embaddings.permute(0, 2, 1, 3)
    
    return patch_embaddings

def to_embaddings(context):
    b, embadding_size, num_heads, head_size = context.shape
    new_shape = context.size()[:-2] + (num_heads*head_size,)
    context = context.view(new_shape)
    
    return context

def Attention(patch_embaddings):
    b, patch_num, embadding_size = patch_embaddings.shape
    num_heads = 12
    head_size = int(embadding_size/num_heads)
    all_head_size = num_heads * head_size
    
    qeury = nn.Linear(embadding_size, all_head_size, bias = True)
    key = nn.Linear(embadding_size, all_head_size, bias = True)
    value = nn.Linear(embadding_size, all_head_size, bias = True)
    
    mixed_key = key(patch_embaddings)
    mixed_value = value(patch_embaddings)
    mixed_query = qeury(patch_embaddings)
    
    key_layer = to_score(mixed_key)
    value_layer = to_score(mixed_value)
    query_layer = to_score(mixed_query)
    
    scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    scores = scores/math.sqrt(head_size)
    
    softmax_scores = nn.functional.softmax(scores, dim = -1)
    
    context = torch.matmul(softmax_scores, value_layer)
    context = context.permute(0, 2, 1, 3)
    context = context.contiguous()
    context = to_embaddings(context)
    
    return context
    
def main():
    img_size = 256
    data = get_datasets(img_size)
    dataloader = DataLoader(dataset = data, batch_size = 5)

    for batch in dataloader:
        img, label = batch
        high_freq_img = HighPass(img)
        patch = PatchEmbadding(high_freq_img, 16)
        Attention(patch)
        

if __name__ == '__main__':
    main()