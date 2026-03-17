import torch
import torch.nn.functional as F


def create_causal_mask(seq_len):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)


def cross_attention(encoder_out, decoder_state):
    d_model = encoder_out.shape[-1]
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)
    W_v = torch.randn(d_model, d_model)
    Q = decoder_state @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v
    scores = (Q @ K.transpose(-2, -1)) / (d_model ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


def feed_forward(x, W1, b1, W2, b2):
    return F.relu(x @ W1 + b1) @ W2 + b2


def add_and_norm(x, sublayer_out):
    return F.layer_norm(x + sublayer_out, x.shape[-1:])


def init_encoder_params(d_model, d_ff):
    return {
        "W_q": torch.randn(d_model, d_model),
        "W_k": torch.randn(d_model, d_model),
        "W_v": torch.randn(d_model, d_model),
        "W1": torch.randn(d_model, d_ff),
        "b1": torch.zeros(d_ff),
        "W2": torch.randn(d_ff, d_model),
        "b2": torch.zeros(d_model),
    }


def encoder_block(x, params):
    Q = x @ params["W_q"]
    K = x @ params["W_k"]
    V = x @ params["W_v"]
    attn_out, _ = scaled_dot_product_attention(Q, K, V)
    x = add_and_norm(x, attn_out)
    ffn_out = feed_forward(x, params["W1"], params["b1"], params["W2"], params["b2"])
    return add_and_norm(x, ffn_out)


def init_decoder_params(d_model, d_ff, vocab_size):
    return {
        "W_q1": torch.randn(d_model, d_model),
        "W_k1": torch.randn(d_model, d_model),
        "W_v1": torch.randn(d_model, d_model),
        "W_q2": torch.randn(d_model, d_model),
        "W_k2": torch.randn(d_model, d_model),
        "W_v2": torch.randn(d_model, d_model),
        "W1": torch.randn(d_model, d_ff),
        "b1": torch.zeros(d_ff),
        "W2": torch.randn(d_ff, d_model),
        "b2": torch.zeros(d_model),
        "W_proj": torch.randn(d_model, vocab_size),
    }


def decoder_block(y, Z, params):
    seq_len = y.shape[-2]
    mask = create_causal_mask(seq_len)

    Q = y @ params["W_q1"]
    K = y @ params["W_k1"]
    V = y @ params["W_v1"]
    masked_attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
    y = add_and_norm(y, masked_attn_out)

    cross_out, _ = cross_attention(Z, y)
    y = add_and_norm(y, cross_out)

    ffn_out = feed_forward(y, params["W1"], params["b1"], params["W2"], params["b2"])
    return add_and_norm(y, ffn_out)


if __name__ == "__main__":
    torch.manual_seed(7)

    d_model = 512
    d_ff = d_model * 4
    vocab_size = 100
    num_layers = 6

    VOCAB = [f"word_{i}" for i in range(vocab_size - 2)] + ["<START>", "<EOS>"]
    START_IDX = vocab_size - 2
    EOS_IDX = vocab_size - 1

    enc_layers = [init_encoder_params(d_model, d_ff) for _ in range(num_layers)]
    dec_layers = [init_decoder_params(d_model, d_ff, vocab_size) for _ in range(num_layers)]

    encoder_input = torch.randn(1, 2, d_model)
    print("Encoder input shape (simulating 'Thinking Machines'):", encoder_input.shape)

    Z = encoder_input
    for params in enc_layers:
        Z = encoder_block(Z, params)
    print("Encoder output Z shape:", Z.shape)

    embedding_table = torch.randn(vocab_size, d_model)
    current_ids = [START_IDX]
    max_steps = 20

    print(f"\nStarting generation: {[VOCAB[i] for i in current_ids]}")

    while len(current_ids) < max_steps:
        y = embedding_table[current_ids].unsqueeze(0)
        for params in dec_layers:
            y = decoder_block(y, Z, params)
        logits = y[:, -1, :] @ dec_layers[-1]["W_proj"]
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.argmax(probs, dim=-1).item()
        current_ids.append(next_idx)
        print(f"Step {len(current_ids) - 1}: '{VOCAB[next_idx]}'")
        if next_idx == EOS_IDX:
            break

    print("\nGenerated sequence:", " ".join(VOCAB[i] for i in current_ids))
