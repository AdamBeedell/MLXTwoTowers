import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import logging
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import utils
import random

min_freq = 10
context_size = 2
embed_dim = 300
epochs = 5
learning_rate = 0.01
patience = 10000

parser = argparse.ArgumentParser(
    description="Train skipgram word2vec model with negative sampling."
)
parser.add_argument(
    "--corpus", default="data/text8", help="Input text file for training"
)
parser.add_argument(
    "--model",
    default="data/word2vec_skipgram.pth",
    help="Output file to save embeddings",
)
parser.add_argument("--batch_size", default=8192, type=int, help="Batch size")
args = parser.parse_args()

input_file = args.corpus
outfile = args.model
batch_size = args.batch_size
utils.setup_logging()


class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        center, context = self.data[idx]  #
        return torch.tensor(center, dtype=torch.long), torch.tensor(
            context, dtype=torch.long
        )


# === Build dataset ===
def make_skipgram_dataset(indices):
    dataset = []
    for i in range(context_size, len(indices) - context_size):
        center_word = indices[i]
        # Dynamic window - randomly choose smaller windows
        dynamic_window = random.randint(1, context_size)  # 1 or 2 instead of always 2
        for j in range(i - dynamic_window, i + dynamic_window + 1):
            if j != i and 0 <= j < len(indices):
                context_word = indices[j]
                dataset.append((center_word, context_word))
    return dataset


# Add this to reduce very common words like "the", "and"
def subsample_frequent_words(indices, word_counts, threshold=1e-3):
    total_words = sum(word_counts.values())
    subsampled = []
    for word_idx in indices:
        word_freq = word_counts[word_idx] / total_words
        prob = (threshold / word_freq) ** 0.5
        if random.random() < prob:
            subsampled.append(word_idx)
    return subsampled


# === Model ===
class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center_words, context_words, neg_samples):
        # Center word embeddings
        center_embeds = self.in_embed(center_words)  # [batch, embed_dim]

        # Context word embeddings
        context_embeds = self.out_embed(context_words)  # [batch, embed_dim]
        neg_embeds = self.out_embed(neg_samples)  # [batch, num_neg, embed_dim]

        # Dot products
        pos_scores = (center_embeds * context_embeds).sum(dim=1)  # [batch]
        neg_scores = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(
            2
        )  # [batch, num_neg]

        # Loss
        pos_loss = F.logsigmoid(pos_scores)
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


def get_negative_samples(target_batch, vocab_size, num_negative=10):
    samp_batch_size = target_batch.size(0)
    return torch.randint(
        0, vocab_size, (samp_batch_size, num_negative), device=target_batch.device
    )


def main():
    device = utils.get_device()
    logging.info(f"Using device: {device}")

    # === Load ===
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().split()

    # === Build vocab ===
    counter = Counter(text)
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    word_to_ix = {word: i for i, word in enumerate(sorted(vocab))}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    vocab_size = len(word_to_ix)

    logging.info(f"Vocab size: {vocab_size}")

    # === Convert text to indices ===
    indices = [word_to_ix[word] for word in text if word in word_to_ix]
    index_counts = Counter(indices)
    indices = subsample_frequent_words(indices, index_counts, threshold=1e-3)
    logging.info(
        f"After subsampling: {len(indices)} tokens (was {len([word_to_ix[word] for word in text if word in word_to_ix])})"
    )

    dataset = make_skipgram_dataset(indices)
    word2vec_dataset = SkipGramDataset(dataset)
    pin_memory = device.type == "cuda"

    model = SkipGramNegativeSampling(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    no_improve_count = 0

    data_loader = DataLoader(
        word2vec_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )

    logging.info(f"Dataset size: {len(dataset)} batch_size {batch_size}")
    start_time = time.time()
    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=100,
            unit="batch",
        )
        total_samples = 0
        for i, (center_batch, context_batch) in pbar:
            center_batch = center_batch.to(device, non_blocking=True)
            context_batch = context_batch.to(device, non_blocking=True)
            neg_samples = get_negative_samples(
                center_batch, vocab_size, num_negative=10
            )

            optimizer.zero_grad()
            loss = model(center_batch, context_batch, neg_samples)  # Swapped order

            loss.backward()
            optimizer.step()

            total_samples += len(center_batch)
            if i % 10 == 0:
                pbar.set_postfix(
                    {
                        "samples": total_samples,
                        "loss": f"{loss.item():.3f}",
                    }
                )
        pure_training_time = time.time() - start_time
        samples_per_sec = total_samples / pure_training_time
        logging.info(
            f"Total training time at batch size {batch_size}: {pure_training_time:.1f} samples/sec {samples_per_sec:.1f}"
        )
        torch.save(
            {
                "embeddings": model.in_embed.weight.data.cpu(),
                "word_to_ix": word_to_ix,
                "ix_to_word": ix_to_word,
            },
            outfile,
        )
    logging.info(f"✅ Embeddings saved to {outfile}")


if __name__ == "__main__":
    main()
