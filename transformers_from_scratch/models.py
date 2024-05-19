from os import fork
import torch
from torch import nn
from torch.nn import functional as F
import torch.jit as jit

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        self._vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(
            num_embeddings=self._vocab_size,
            embedding_dim=self._vocab_size
        )

    def forward(self, idx: int, targets=None): 
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
            
        batch, time, channel = logits.shape


        batch_x_time = batch * time
        logits_reshaped = logits.view(batch_x_time, channel)
        targets_reshaped = targets.view(batch_x_time)
        loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            logits_last_timestep = logits[:,-1,:]
            probs = F.softmax(logits_last_timestep, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class TransformerLanguageModel(nn.Module):
    def __init__(self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        head_size: int,
        context_length: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._head_size = head_size
        self._context_length = context_length

        self.token_embedding_table = nn.Embedding(
            num_embeddings=self._vocab_size,
            embedding_dim=self._embedding_dim
        )
        self.token_position_embedding_table = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )
        
        self.transformer_blocks = nn.Sequential(
            *(
                [
                    TransformerBlock(embedding_dim, context_length, num_heads, dropout)
                    for _ in range(num_layers)
                ] + [nn.LayerNorm(embedding_dim)]
            )
        )
                
        self.language_model_head_linear_layer = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

        

    # @torch.autocast(device_type="cpu")
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None): 
        _, time = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        token_position_embeddings = self.token_position_embedding_table(
            torch.arange(time,)
        )

        x = token_embeddings + token_position_embeddings
        x = self.transformer_blocks(x)
        
        logits = self.language_model_head_linear_layer(x)

        if targets is None:
            return logits, None
            
        batch, time, channel = logits.shape


        batch_x_time = batch * time
        logits_reshaped = logits.view(batch_x_time, channel)
        targets_reshaped = targets.view(batch_x_time)
        loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)

        return logits, loss

    @jit.export
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self._context_length:]
            logits, loss = self(idx_crop, None)

            logits_last_timestep = logits[:,-1,:]
            probs = F.softmax(logits_last_timestep, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        num_heads: int,
        dropout: float,
     ) -> None:
        super().__init__()

        self.multi_head_attention_layer = MultiHeadAttention(
            num_heads,
            embedding_dim // num_heads,
            embedding_dim,
            context_length,
            dropout,
        )

        self.feed_forward_network = FeedForwardNetwork(embedding_dim, dropout)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

    # @torch.autocast(device_type="cpu")
    def forward(self, x):
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))
        x = x + self.feed_forward_network(self.layer_norm_2(x))
        
        return x        
    
class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        num_heads: int,
        head_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList([
            AttentionHead(
               head_size,
               embedding_dim,
               context_length,
               dropout,
           )
           for _ in range(num_heads)
        ])

        self.projection_layer = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )
        self.dropout_layer = nn.Dropout(dropout)

    # @torch.autocast(device_type="cpu")
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out

class AttentionHead(nn.Module):
    def __init__(
        self,
        head_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self._head_size = head_size
        self._embedding_dim = embedding_dim
        self._context_length = context_length

        self.key_layer = nn.Linear(in_features=embedding_dim, out_features=head_size, bias=False)
        self.query_layer = nn.Linear(in_features=embedding_dim, out_features=head_size, bias=False)
        self.value_layer = nn.Linear(in_features=embedding_dim, out_features=head_size, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                   (context_length, context_length)
               )
           )
        )

    # @torch.autocast(device_type="cpu")
    def forward(self, x: torch.Tensor):
        _, time, _ = x.shape # Batch Size, Context Length, Head Size

        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)
    
        weights = (q @ k.transpose(-2, -1)) * self._head_size**-0.5
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)

        out = weights @ v
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
            nn.Dropout(dropout),
        )

    # @torch.autocast(device_type="cpu")
    def forward(self, x):
        return self.ffn(x)

