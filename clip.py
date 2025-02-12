import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.positional_embedding = nn.Parameter(torch.zeros((n_token, n_embed)))

    def forward(self, tokens):

        x = self.token_embedding(tokens)
        x += self.positional_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4*n_embed)
        self.linear_2 = nn.Linear(4*n_embed, n_embed)

    def forward(self, x):

        residue = x

        x = self.layernorm_1(x)

        x = self.attention(x, casual_mask=True)

        x += residue

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residue

        return x
    
class CLIP(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_token: int, n_head: int, n_layers: int):
        super().__init__()

        self.embedding = CLIPEmbedding(n_vocab, n_embed, n_token)
        
        self.layers = nn.ModuleList([
            CLIPLayer(n_head, n_embed) 
            for _ in range(n_layers)])
        
        self.layernorm = nn.LayerNorm(n_embed)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output



