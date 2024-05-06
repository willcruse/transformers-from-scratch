from pathlib import Path

import torch

from transformers_from_scratch.models import BigramLanguageModel
from transformers_from_scratch.tokenizer import CharacterTokenizer

BATCH_SIZE = 32
CONTEXT_LENGTH = 8
TORCH_SEED = 1337

torch.manual_seed(TORCH_SEED)

def read_input_file(file: str) -> str:
    filepath = Path(file)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found at {file}")

    if not filepath.is_file():
        raise FileExistsError(f"File exists at {file} but is not a file")

    return filepath.read_text()

def get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = len(data)
    start_idx = torch.randint(high=data_len - CONTEXT_LENGTH, size=(BATCH_SIZE,))

    x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in start_idx])
    y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in start_idx])
    return (x, y,)

def main(input_text_file: str) -> None:
    input_text = read_input_file(input_text_file)
    
    tokenizer = CharacterTokenizer()
    tokenizer.train(input_text)

    encoded_text = torch.tensor(tokenizer.encode(input_text))

    train_split_pct = 0.9
    train_split_idx = int(len(encoded_text)*train_split_pct)

    train_data = encoded_text[:train_split_idx]
    valid_data = encoded_text[train_split_idx:]

    bigram_model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.Adam(params=bigram_model.parameters(), lr=1e-3)

    for step_num in range(10000):
        x_batch, y_batch = get_batch(train_data)

        logits, loss = bigram_model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_num % 100 == 0:
            print(f"Loss: {round(loss.item(), 3)}")
    
    tokens = bigram_model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()
    decoded = tokenizer.decode(tokens)
    print(decoded)
    
if __name__ == "__main__":
    main("tinyshakespeare.txt")
