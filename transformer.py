import os
import importlib.util
import torch
import torch.nn as nn

_lab4_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4", "transformer.py")
_spec = importlib.util.spec_from_file_location("lab4_transformer", _lab4_path)
_lab4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lab4)

scaled_dot_product_attention = _lab4.scaled_dot_product_attention
feed_forward = _lab4.feed_forward


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.shape[0]

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return feed_forward(x, self.linear1.weight.T, self.linear1.bias,
                            self.linear2.weight.T, self.linear2.bias)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, src_mask))
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y, encoder_out, tgt_mask=None, src_mask=None):
        y = self.norm1(y + self.self_attn(y, y, y, tgt_mask))
        y = self.norm2(y + self.cross_attn(y, encoder_out, encoder_out, src_mask))
        y = self.norm3(y + self.ffn(y))
        return y


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, pad_idx):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_out, tgt_mask=None, src_mask=None):
        y = self.tgt_embedding(tgt)
        for layer in self.decoder_layers:
            y = layer(y, encoder_out, tgt_mask, src_mask)
        return y

    def forward(self, src, tgt):
        seq_len = tgt.shape[1]
        tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(tgt.device)

        encoder_out = self.encode(src)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask)
        return self.projection(decoder_out)

    def autoregressive_decode(self, src, start_idx, end_idx, max_len=50):
        self.eval()
        with torch.no_grad():
            encoder_out = self.encode(src)
            generated = [start_idx]

            for _ in range(max_len):
                tgt = torch.tensor([generated], dtype=torch.long)
                seq_len = tgt.shape[1]
                tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
                decoder_out = self.decode(tgt, encoder_out, tgt_mask)
                logits = self.projection(decoder_out[:, -1, :])
                next_idx = torch.argmax(logits, dim=-1).item()
                generated.append(next_idx)
                if next_idx == end_idx:
                    break

        return generated
