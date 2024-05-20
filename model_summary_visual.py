import torch
from torch import nn
from models import build_model
from hubconf import detr_resnet50
from models.transformer import TransformerEncoderWithPrompt, TransformerEncoderLayer
from torchsummary import summary



d_model=512
nhead=8
dim_feedforward=2048
dropout=0.1
activation="relu"
normalize_before=False

encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

# src, src_mask, src_key_padding_mask, pos, prompt


# Create dummy data
src =  torch.randn([1,100,512])
pos = None

#G-prompt
g_prompt_shape = [1, 2, 2, nhead, d_model//nhead]
g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
nn.init.uniform_(g_prompt, -1, 1)

out = encoder_layer(src,None,None,None,g_prompt)
print(out.shape)