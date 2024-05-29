import torch
from torch import nn
import math
from src.Configueration import VitConfig

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