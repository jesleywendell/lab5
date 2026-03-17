import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


TOKENIZER_NAME = "bert-base-multilingual-cased"
DATASET_NAME = "bentrevett/multi30k"
SUBSET_SIZE = 1000
PAD_IDX = 0


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def load_pairs(subset_size=SUBSET_SIZE):
    ds = load_dataset(DATASET_NAME, split="train")
    pairs = [(ds[i]["en"], ds[i]["de"]) for i in range(subset_size)]
    return pairs


def tokenize_pairs(pairs, tokenizer, max_len=64):
    src_sequences = []
    tgt_sequences = []

    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id

    for src_text, tgt_text in pairs:
        src_ids = tokenizer.encode(src_text, add_special_tokens=False)
        tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)

        src_ids = src_ids[:max_len]
        tgt_ids = tgt_ids[:max_len - 2]

        tgt_ids = [bos_id] + tgt_ids + [eos_id]

        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)

    return src_sequences, tgt_sequences


def pad_sequence(seq, length, pad_idx=PAD_IDX):
    return seq + [pad_idx] * (length - len(seq))


def build_tensors(src_sequences, tgt_sequences):
    src_len = max(len(s) for s in src_sequences)
    tgt_len = max(len(t) for t in tgt_sequences)

    src_padded = [pad_sequence(s, src_len) for s in src_sequences]
    tgt_padded = [pad_sequence(t, tgt_len) for t in tgt_sequences]

    src_tensor = torch.tensor(src_padded, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long)

    return src_tensor, tgt_tensor


class TranslationDataset(Dataset):
    def __init__(self, src_tensor, tgt_tensor):
        self.src = src_tensor
        self.tgt = tgt_tensor

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def build_dataloader(batch_size=32):
    tokenizer = load_tokenizer()
    pairs = load_pairs()
    src_seqs, tgt_seqs = tokenize_pairs(pairs, tokenizer)
    src_tensor, tgt_tensor = build_tensors(src_seqs, tgt_seqs)
    dataset = TranslationDataset(src_tensor, tgt_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, tokenizer
