import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights


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

        self.attention = ScaledDotProductAttention()

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

        out, _ = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


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
