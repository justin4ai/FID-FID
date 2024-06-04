import torch
from torch import nn
import math
from src.Configueration import VitConfig
from src.CustomDataLoader import DataProcesser

# def show_image(img_tensor):
#     printer = DataProcesser()
#     printer.show_tensor_image(img_tensor)

class PatchEmbadding(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.num_channels = config.num_channels
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_patchs
        self.hidden_size = config.hidden_size
        self.device = config.device
        
    def forward(self, batch):
        # show_image(batch)
        b, c, h, w = batch.shape
        hidden_states = torch.zeros((b, self.num_patches, self.hidden_size), device = self.device)
        patch_len = int(h/self.patch_size)
        for b_idx in range(b):
            for i in range(patch_len):
                for j in range(patch_len):
                    hidden_states[b_idx, i*patch_len+j, :] = batch[b_idx, :, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size].flatten()
        
        return hidden_states

class HighPass(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.num_channels = config.num_channels
        self.image_size = config.image_size
        self.highpass_rate = config.highpass_rate
        
    def highpass(self, batch):
        b, c, h, w = batch.shape
        mid = h//2
        rate = h//self.highpass_rate
        fft_image = torch.fft.fftshift(torch.fft.fft2(batch, dim = [-2, -1], norm = 'ortho'))
        fft_image[:, :, mid-rate:mid+rate, mid-rate:mid+rate] = 0.0
        highpassed = torch.fft.ifft2(torch.fft.ifftshift(fft_image), dim = [-2, -1], norm = 'ortho')
        highpassed = highpassed.real
        
        return highpassed
        
    def forward(self, batch):
        # show_image(batch[0:14])
        highpassed = self.highpass(batch)
        # show_image(highpassed[0:14])
        return highpassed

class LocalHighPass(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.device = config.device
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.num_patchs = config.num_patchs
        self.highpass = HighPass()
           
    def forward(self, hidden_states):
        batch_num = hidden_states.shape[0]
        reshape = torch.zeros((self.num_patchs, self.num_channels, self.patch_size, self.patch_size), device = self.device)
        for b_idx in range(batch_num):
            for p_idx in range(self.num_patchs):
                reshape[p_idx, :, :, :] = hidden_states[b_idx, p_idx, :].reshape(shape = (self.num_channels, self.patch_size, self.patch_size))

            hidden_states[b_idx, :, :] = self.highpass(reshape).flatten(1)
    
        return hidden_states

class MSAttention(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_patchs = config.num_patchs
        self.num_attention_heads = config.num_attention_heads
        self.head_size = int(self.hidden_size/self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_size
        
        self.position_embaddings = nn.Parameter(torch.randn(1, self.num_patchs, self.hidden_size))
        
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
    
    def forward(self, hidden_states, first_block = False):
        if first_block:
            hidden_states = hidden_states + self.position_embaddings
            
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
    def __init__(self, config = VitConfig(), first_block = False):
        super().__init__()
        self.highpass = LocalHighPass()
        self.attention = MSAttention()
        self.mlp = MLP()
        self.layernorm_before = nn.LayerNorm(config.hidden_size)
        self.layernorm_att = nn.LayerNorm(config.hidden_size)
        self.layernorm_mlp = nn.LayerNorm(config.hidden_size)
        self.first_block = first_block
        
    def forward(self, hidden_states):
        hidden_states = self.highpass(hidden_states)
        self_attention_output = self.attention(self.layernorm_before(hidden_states), self.first_block)
        hidden_states = hidden_states + self.layernorm_att(self_attention_output)
        
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + self.layernorm_mlp(mlp_output)
        
        return hidden_states

class VitEncoder(nn.Module):
    def __init__(self, config = VitConfig()):
        super().__init__()
        self.patch_embaddings = PatchEmbadding()
        self.blocks = nn.ModuleList([])
        self.blocks.append(EncoderBlock(first_block = True))
        for _ in range(config.num_hidden_layers):
            block = EncoderBlock()
            self.blocks.append(block)
            
    def forward(self, batch):
        x = self.patch_embaddings(batch)
        for block in self.blocks:
            x = block(x)
        
        return x