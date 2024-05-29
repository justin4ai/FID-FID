import torch

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