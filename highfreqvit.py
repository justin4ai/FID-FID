import glob
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel
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

def tensor_to_image(tensor):
    reverse_transforms = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage()
    ])
    
    return reverse_transforms(tensor)

def show_tensor_image(t_img): 
    if len(t_img.shape) == 4:
        t_img = t_img[0, :, : ,:]
    
    img = tensor_to_image(t_img)
    plt.imshow(img)
    plt.show()

def get_datasets(size = 224):
    trans = transforms.Compose([transforms.Resize((size, size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return MyDatasets(path = "./datasets", transform = trans)

class VitConfig():
    def __init__(self,
                 num_hidden_layers = 12,
                 image_size = 224,
                 num_channels = 3,
                 patch_size = 16,
                 highpass_rate = 8,
                 num_attention_heads = 12,
                 intermediate_size = 3072
        ):
        self.num_hidden_layers = num_hidden_layers
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patchs = (self.image_size//self.patch_size) ** 2
        self.hidden_size = self.num_channels * (self.patch_size ** 2)
        self.highpass_rate = highpass_rate
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

class HighPass(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.num_channels = config.num_channels
        self.image_size = config.image_size
        self.highpass_rate = config.highpass_rate
        
    def highpass_filter(self, batch):
        mid = self.image_size//2
        rate = self.image_size//self.highpass_rate
        fft_image = torch.fft.fftshift(torch.fft.fft2(batch, dim = [-2, -1], norm = 'ortho'))
        fft_image[:, :, mid-rate:mid+rate, mid-rate:mid+rate] = 0.0
        highpassed = torch.fft.ifft2(torch.fft.ifftshift(fft_image), dim = [-2, -1], norm = 'ortho')
        highpassed = highpassed.real
        
        return highpassed
        
    def forward(self, batch):
        highpassed = self.highpass_filter(batch)
        return highpassed

class PatchEmbadding(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.num_channels = config.num_channels
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_patchs
        self.hidden_size = config.hidden_size

        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size = self.patch_size, stride = self.patch_size)
        self.position_embaddings = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_size))
        
    def forward(self, batch):
        hidden_states = self.projection(batch).flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embaddings
        
        return hidden_states

class MSAttention(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_patchs = config.num_patchs
        self.num_attention_heads = config.num_attention_heads
        self.head_size = int(self.hidden_size/self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_size
        
        self.qeury = nn.Linear(self.hidden_size, self.all_head_size, bias = True)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias = True)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias = True)

    def to_score(self, mixed):
        new_shape = mixed.size()[:-1] + (self.num_attention_heads, self.head_size)
        mixed_to_score = mixed.view(new_shape)
        mixed_to_score = mixed_to_score.permute(0, 2, 1, 3)
        
        return mixed_to_score

    def to_embaddings(self, context):
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.hidden_size,)
        context_to_embaddings = context.view(new_shape)
        
        return context_to_embaddings
    
    def forward(self, hidden_states):
        mixed_key = self.key(hidden_states)
        mixed_value = self.value(hidden_states)
        mixed_query = self.qeury(hidden_states)
        
        key_layer = self.to_score(mixed_key)
        value_layer = self.to_score(mixed_value)
        query_layer = self.to_score(mixed_query)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores/math.sqrt(self.head_size)
        softmax_scores = nn.functional.softmax(scores, dim = -1)
        
        context = torch.matmul(softmax_scores, value_layer)
        outputs = self.to_embaddings(context)
        
        return outputs

class MLP(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.dense_before = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.ReLU()
        self.dense_after = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, x):
        x = self.dense_before(x)
        x = self.activation(x)
        x = self.dense_after(x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.attention = MSAttention()
        self.mlp = MLP()
        self.layernorm_before = nn.LayerNorm(config.hidden_size)
        self.layernorm_att = nn.LayerNorm(config.hidden_size)
        self.layernorm_mlp = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states):
        self_attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = hidden_states + self.layernorm_att(self_attention_output)
        
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + self.layernorm_mlp(mlp_output)
        
        return hidden_states

class VitEncoder(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.high_pass_filter = HighPass()
        self.patch_embaddings = PatchEmbadding()
        self.blocks = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            block = EncoderBlock()
            self.blocks.append(block)
            
    def forward(self, batch):
        x = self.patch_embaddings(self.high_pass_filter(batch))
        for block in self.blocks:
            x = block(x)
        
        return x

class CrossAttention(MSAttention):
    def __init__(self, config = VitConfig()):
        super().__init__()
    
    def forward(self, query_hidden_states, key_value_hidden_states):
        mixed_key = self.key(key_value_hidden_states)
        mixed_value = self.value(key_value_hidden_states)
        mixed_query = self.qeury(query_hidden_states)
        
        key_layer = self.to_score(mixed_key)
        value_layer = self.to_score(mixed_value)
        query_layer = self.to_score(mixed_query)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores/math.sqrt(self.head_size)
        softmax_scores = nn.functional.softmax(scores, dim = -1)
        
        context = torch.matmul(softmax_scores, value_layer)
        outputs = self.to_embaddings(context)
        
        return outputs

class CrossAttentionBlock(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__() 
        self.cross_attention = CrossAttention()
        self.mlp = MLP()
        self.layernorm_att = nn.LayerNorm(config.hidden_size)
        self.layernorm_mlp = nn.LayerNorm(config.hidden_size)
        
    def forward(self, image_hidden_states, freq_hidden_states):
        cross_attention_output = self.cross_attention(freq_hidden_states, image_hidden_states)
        hidden_states = freq_hidden_states + self.layernorm_att(cross_attention_output)
        
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + self.layernorm_mlp(mlp_output)
        
        return hidden_states

class HighFreqVitEncoder(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.original_img_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.highfreq_img_encoder = VitEncoder()
        self.CrossBlocks = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            block = CrossAttentionBlock()
            self.CrossBlocks.append(block)
            
    def forward(self, batch):
        image_embaddings = self.original_img_encoder(batch).last_hidden_state
        image_embaddings = image_embaddings[:,1:,:]
        
        x = self.highfreq_img_encoder(batch)
        for block in self.CrossBlocks:
            x = block(image_embaddings, x)
        
        return x


def main():
    data = get_datasets()
    dataloader = DataLoader(dataset = data, batch_size = 5)
    highfreq_encoder = HighFreqVitEncoder()
    
    for batch in dataloader:
        img, label = batch
        embaddings = highfreq_encoder(img)
        print(embaddings.shape)


if __name__ == '__main__':
    main()