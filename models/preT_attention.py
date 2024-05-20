import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None ,prompt=None):
        batch_size, query_len, embed_dim = query.size()
        _, key_len, _ = key.size()

        # Project and reshape query, key, value
        query = self.query_projection(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_projection(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_projection(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        

        if prompt is not None:
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads
            key = torch.cat([key_prefix, key], dim=2)
            value = torch.cat([value_prefix, value], dim=2)            


        # Compute scaled dot product attention
        scaling = float(self.head_dim) ** -0.5
        query = query * scaling
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
            
        # Apply attention mask
        # if attn_mask is not None:
        #     attention_scores = attention_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply the key padding mask
        # if key_padding_mask is not None:
        #     key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # Reshape to (batch_size, 1, 1, key_len)
        #     attention_scores = attention_scores.masked_fill(key_padding_mask, float('-inf'))

        # Softmax to obtain attention probabilities
        attention_probs = self.softmax(attention_scores)
        attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)

        # Apply attention to the value
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)

        # Final linear layer
        output = self.out_projection(context)
        return output