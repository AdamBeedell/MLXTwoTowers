import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import logging
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import argparse

import utils

min_freq = 10
context_size = 2
embed_dim = 300
batch_size = 512
epochs = 5
learning_rate = 0.01
patience = 10000

parser = argparse.ArgumentParser(
    description="Train CBOW word2vec model with negative sampling."
)
parser.add_argument("--corpus", required=True, help="Input text file for training")
parser.add_argument("--model", required=True, help="Output file to save embeddings")
args = parser.parse_args()

input_file = args.corpus
outfile = args.model

utils.setup_logging()


class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        center, context = self.data[
            idx
        ]  # Now center, context instead of context, target
        return torch.tensor(center, dtype=torch.long), torch.tensor(
            context, dtype=torch.long
        )


# === Build dataset ===
def make_skipgram_dataset(indices):
    dataset = []
    for i in range(context_size, len(indices) - context_size):
        center_word = indices[i]
        # Generate all context words within window
        for j in range(i - context_size, i + context_size + 1):
            if j != i:  # Skip the center word itself
                context_word = indices[j]
                dataset.append((center_word, context_word))
    return dataset


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

    dataset = make_skipgram_dataset(indices)
    word2vec_dataset = SkipGramDataset(dataset)
    pin_memory = device.type == "cuda"
    data_loader = DataLoader(
        word2vec_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )

    logging.info(f"Dataset size: {len(dataset)}")

    model = SkipGramNegativeSampling(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    no_improve_count = 0
    for epoch in range(epochs):
        for i, (center_batch, context_batch) in enumerate(data_loader):  # Swapped order
            center_batch = center_batch.to(device, non_blocking=True)
            context_batch = context_batch.to(device, non_blocking=True)
            neg_samples = get_negative_samples(
                center_batch, vocab_size, num_negative=10
            )

            optimizer.zero_grad()
            loss = model(center_batch, context_batch, neg_samples)  # Swapped order

            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improve_count = 0
                torch.save(
                    {
                        "embeddings": model.in_embed.weight.data.cpu(),
                        "word_to_ix": word_to_ix,
                        "ix_to_word": ix_to_word,
                    },
                    outfile,
                )
            else:
                no_improve_count += 1
            if no_improve_count > patience:
                logging.info(f"Early stopping at epoch {epoch + 1}, step {i}")
                break
            if i % 1000 == 999:
                logging.info(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")
    logging.info(f"✅ Embeddings saved to {outfile}")


if __name__ == "__main__":
    main()
