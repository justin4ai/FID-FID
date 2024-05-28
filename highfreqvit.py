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
import random
from tqdm import tqdm

class MyDatasets(Dataset):
    def __init__(self, path, train = True, transform = None):
        self.path = path
        
        if train:
            self.data_real = glob.glob(self.path + '/train/real/*.jpg')
            self.data_real = random.sample(self.data_real, 1000)
            self.data_generated = glob.glob(self.path + '/train/generated/*.*')
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
                 intermediate_size = 3072,
                 num_labels = 2,
                 labels = ["real", "generated"]
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
        self.num_labels = num_labels
        self.labels = labels
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

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
        self.hidden_size = config.hidden_size
        self.num_patchs = config.num_patchs
        self.num_attention_heads = config.num_attention_heads
        self.head_size = int(self.hidden_size/self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_size
        
        self.qeury = nn.Linear(self.hidden_size, self.all_head_size, bias = True)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias = True)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias = True)

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
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.CrossBlocks = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            block = CrossAttentionBlock()
            self.CrossBlocks.append(block)
            
    def forward(self, batch):
        image_embaddings = self.original_img_encoder(batch).last_hidden_state
        cls_tokens = self.cls_token.expand(batch.shape[0], -1, -1)
        x = torch.cat((cls_tokens, self.highfreq_img_encoder(batch)), dim = 1)
        
        for block in self.CrossBlocks:
            x = block(image_embaddings, x)
        
        return x

class HighFreqVitClassifier(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.vit = HighFreqVitEncoder()
        self.num_labels = config.num_labels
        self.class_name = config.labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_func = nn.CrossEntropyLoss()
        self.output = nn.Softmax(dim = -1)
       
    def one_hot_encoding(self, labels):
        one_hot_vectors = []
        for target_class in labels:
            vector = np.zeros(self.num_labels)
            for idx in range(self.num_labels):  
                if target_class == self.class_name[idx]:
                    vector[idx] = 1
            one_hot_vectors.append(vector)
            
        return torch.as_tensor(np.array(one_hot_vectors))
        
    def reverse_one_hot_encoding(self, outputs):
        labels = []
        for vector in outputs:
            if vector[0] == 1:
                labels.append('real')
            else:
                labels.append('generated')
        
        return labels
    
    def forward(self, image, labels, device):
        labels = self.one_hot_encoding(labels).to(device)
        vit_output = self.vit(image)
        logits = self.classifier(vit_output[:, 0, :])
        loss = self.loss_func(logits, labels)

        logit_outputs = self.output(logits)
        logit_outputs[logit_outputs>=0.5] = 1
        logit_outputs[logit_outputs<0.5] = 0
        logit_outputs = self.reverse_one_hot_encoding(logit_outputs)
        return (logit_outputs, loss)

def main():
    data = get_datasets()
    batch_size = 50
    dataloader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    num_datas = len(dataloader)
    classifier = HighFreqVitClassifier()
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name())
    classifier.to(device)
    num_epochs = 100
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.01)
    checkpoint = None
    
    if glob.glob("./*.pt"):
        checkpoint = torch.load("./model_state_dict.pt")
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    print(checkpoint['epoch'])    
    for epoch in range(num_epochs):
        
        if bool(checkpoint) and (epoch <= checkpoint_epoch):
            continue
        
        correct = 0
        
        for batch_num, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            
            img, labels = batch
            labels = list(labels)
            
            logits, loss = classifier(img.to(device), labels, device)
            
            for idx in range(len(labels)):
                if labels[idx] == logits[idx]:
                    correct += 1
            
            loss.backward()
            optimizer.step()
            
        acc = correct/num_datas
        print("\nEpoch : ", epoch, "\nTraining accuracy : ", acc, "%\nTraining Loss : ", loss.item(), "\n")
        
        torch.save({
                    'epoch' : epoch, 
                    'model_state_dict' : classifier.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss' : loss
                    }, "./model_state_dict.pt")
        

if __name__ == '__main__':
    main()