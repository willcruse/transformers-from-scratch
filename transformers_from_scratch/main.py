from pathlib import Path
from cProfile import Profile
from pstats import SortKey, Stats

import torch

from transformers_from_scratch.models import BigramLanguageModel, TransformerLanguageModel
from transformers_from_scratch.tokenizer import CharacterTokenizer

BATCH_SIZE = 64
CONTEXT_LENGTH = 256
NUM_HEADS = 6
NUM_LAYERS = 6
HEAD_SIZE = 16
DROPOUT = 0.2
EMBEDDINGS_DIMENSION = 384
LEARNING_RATE = 3e-4
MAX_TRAINING_ITERATIONS = 3000
EVAL_INTERVAL = 500
EVAL_ITERATIONS = 200
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
    idxs = torch.randint(high=data_len - CONTEXT_LENGTH, size=(BATCH_SIZE,))

    x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in idxs])
    y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in idxs])
    return (x, y,)

@torch.no_grad()
def estimate_loss(model, train_data: torch.Tensor, validation_data: torch.Tensor):
    model.eval()

    train_losses = torch.zeros(EVAL_ITERATIONS)
    for i in range(EVAL_ITERATIONS):
        x_batch, y_batch = get_batch(train_data)
        _, loss = model(x_batch, y_batch)
        train_losses[i] = loss.item()

    validation_losses = torch.zeros(EVAL_ITERATIONS)
    for i in range(EVAL_ITERATIONS):
        x_batch, y_batch = get_batch(validation_data)
        _, loss = model(x_batch, y_batch)
        validation_losses[i] = loss.item()

    return {
        "train": train_losses.mean(),
        "valid": validation_losses.mean(),
    }
    
def main(input_text_file: str) -> None:
    input_text = read_input_file(input_text_file)
    
    tokenizer = CharacterTokenizer()
    tokenizer.train(input_text)

    encoded_text = torch.tensor(tokenizer.encode(input_text))

    train_split_pct = 0.9
    train_split_idx = int(len(encoded_text)*train_split_pct)

    train_data = encoded_text[:train_split_idx]
    valid_data = encoded_text[train_split_idx:]

    # bigram_model = BigramLanguageModel(tokenizer.vocab_size)
    transformer_model = TransformerLanguageModel(
        tokenizer.vocab_size,
        EMBEDDINGS_DIMENSION,
        NUM_HEADS,
        HEAD_SIZE,
        CONTEXT_LENGTH,
        NUM_LAYERS,
        DROPOUT
    )
    optimizer = torch.optim.Adam(params=transformer_model.parameters(), lr=LEARNING_RATE)
    print("Setup Model\nStarting Training...")

    for step_num in range(MAX_TRAINING_ITERATIONS):
        if step_num % EVAL_INTERVAL == 0 or step_num == MAX_TRAINING_ITERATIONS - 1:
            losses = estimate_loss(transformer_model, train_data, valid_data)
            rounded_train_losses = round(losses["train"].item(), 3)
            rounded_validation_loss = round(losses["valid"].item(), 3)
            print(f"Step: {step_num} Training Loss: {rounded_train_losses} Validation Loss: {rounded_validation_loss}")

    
        x_batch, y_batch = get_batch(train_data)

        logits, loss = transformer_model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    tokens = transformer_model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()
    decoded = tokenizer.decode(tokens)
    print(decoded)

    
if __name__ == "__main__":
    main("tinyshakespeare.txt")
