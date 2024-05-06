import torch
from torch import nn
from torch.nn import functional as F

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
