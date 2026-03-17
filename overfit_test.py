import torch
import torch.nn as nn

from transformer import Transformer
from data import load_tokenizer, load_pairs, tokenize_pairs, build_tensors, PAD_IDX

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = D_MODEL * 4
EPOCHS = 80
LEARNING_RATE = 1e-3
NUM_SAMPLES = 8


def run():
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id

    pairs = load_pairs()[:NUM_SAMPLES]
    src_seqs, tgt_seqs = tokenize_pairs(pairs, tokenizer)
    src_tensor, tgt_tensor = build_tensors(src_seqs, tgt_seqs)

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

    print(f"Overfitting test on {NUM_SAMPLES} samples for {EPOCHS} epochs\n")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        tgt_input = tgt_tensor[:, :-1]
        tgt_target = tgt_tensor[:, 1:]

        logits = model(src_tensor, tgt_input)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {loss.item():.4f}")

    print("\n--- Autoregressive inference on a training sample ---")

    sample_idx = 0
    src_sentence, expected_translation = pairs[sample_idx]

    src_input = src_tensor[sample_idx].unsqueeze(0)
    generated_ids = model.autoregressive_decode(src_input, bos_id, eos_id, max_len=60)
    generated_text = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)

    print(f"Source      : {src_sentence}")
    print(f"Expected    : {expected_translation}")
    print(f"Generated   : {generated_text}")


if __name__ == "__main__":
    run()
