import torch
from torch import nn
import math
import numpy as np
from src.Configueration import VitConfig
from src import FreqEncoder
from transformers import ViTModel

class CrossAttention(FreqEncoder.MSAttention):
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
        self.original_img_encoder.eval() # Freeze pretrained ViT
        self.highfreq_img_encoder = FreqEncoder.VitEncoder()
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
    
    def forward(self, image, labels, device = None):
        labels = self.one_hot_encoding(labels).to(device)
        vit_output = self.vit(image)
        logits = self.classifier(vit_output[:, 0, :])
        loss = self.loss_func(logits, labels)

        logit_outputs = self.output(logits)
        logit_outputs[logit_outputs>=0.5] = 1
        logit_outputs[logit_outputs<0.5] = 0
        logit_outputs = self.reverse_one_hot_encoding(logit_outputs)
        return (logit_outputs, loss)