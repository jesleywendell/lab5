import torch
import torch.nn as nn

from transformer import Transformer
from data import build_dataloader, PAD_IDX

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = D_MODEL * 4
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def train():
    loader, tokenizer = build_dataloader(batch_size=BATCH_SIZE)

    vocab_size = tokenizer.vocab_size

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        pad_idx=PAD_IDX,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        num_batches = 0

        for src, tgt in loader:
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            logits = model(src, tgt_input)

            logits_flat = logits.reshape(-1, vocab_size)
            target_flat = tgt_target.reshape(-1)

            loss = criterion(logits_flat, target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("\nModel saved to model.pt")

    return model, tokenizer


if __name__ == "__main__":
    train()
