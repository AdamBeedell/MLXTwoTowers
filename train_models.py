import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer
import utils
from torch.utils.data import DataLoader

from dataset import DATASET_FILE
from model import Tower
from utils import MODEL_FILE

embed_dim = 256
margin = 0.1
learning_rate = 0.0005
embedding = nn.Embedding(1000, embed_dim)
batch_size = 8192
epochs = 5
dropout_rate = 0.1


class TripletDataLoader(DataLoader):
    def __init__(self, dataset, device):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )


def validate_model(query_tower, doc_tower, validation_dataloader, device):
    query_tower.eval()
    doc_tower.eval()

    total_loss = 0.0
    num_batches = 0
    zero = torch.tensor(0.0).to(device)

    with torch.no_grad():  # Important: no gradients during validation
        for batch in validation_dataloader:
            query, positive, negative = [b.to(device) for b in batch.values()]

            # Forward pass (same as training)
            q = query_tower(query)
            pos = doc_tower(positive)
            neg = doc_tower(negative)

            # Calculate loss (same as training)
            dst_pos = F.cosine_similarity(q, pos)
            dst_neg = F.cosine_similarity(q, neg)
            loss = torch.max(zero, margin - dst_pos + dst_neg).mean()

            total_loss += loss.item()
            num_batches += 1

    query_tower.train()  # Set back to training mode
    doc_tower.train()

    return total_loss / num_batches


def calculate_retrieval_metrics(
    query_tower, doc_tower, validation_dataset, device, k=5
):
    """Calculate retrieval-specific metrics like accuracy@k"""
    query_tower.eval()
    doc_tower.eval()

    correct_at_k = 0
    total_queries = 0

    with torch.no_grad():
        for item in validation_dataset:
            query = item["query"].unsqueeze(0).to(device)
            positive = item["positive"].unsqueeze(0).to(device)
            negative = item["negative"].unsqueeze(0).to(device)

            q_enc = query_tower(query)
            pos_enc = doc_tower(positive)
            neg_enc = doc_tower(negative)

            pos_sim = F.cosine_similarity(q_enc, pos_enc)
            neg_sim = F.cosine_similarity(q_enc, neg_enc)

            # Check if positive document ranks higher than negative
            if pos_sim > neg_sim:
                correct_at_k += 1
            total_queries += 1

    return correct_at_k / total_queries if total_queries > 0 else 0.0


def main():
    utils.setup_logging()
    device = utils.get_device()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_size = tokenizer.vocab_size
    query_tower = Tower(vocab_size, embed_dim, dropout_rate).to(device)
    doc_tower = Tower(vocab_size, embed_dim, dropout_rate).to(device)

    logging.info("Loading datasets")
    datasets = torch.load(DATASET_FILE, weights_only=False)
    train_dataset = datasets["train_triplets"]
    validation_dataset = datasets["val_triplets"]
    test_dataset = datasets["test_triplets"]
    logging.info(
        f"Dataset sizes: training {len(train_dataset)} validation: {len(validation_dataset)} test: {len(test_dataset)}"
    )
    training_dataloader = TripletDataLoader(train_dataset, device)
    validation_dataloader = TripletDataLoader(validation_dataset, device)
    test_dataloader = TripletDataLoader(test_dataset, device)
    params = list(query_tower.parameters()) + list(doc_tower.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    logging.info("Starting training")
    zero = torch.tensor(0.0)
    best_val_loss = float("inf")
    for epoch in range(epochs):
        total_train_loss = 0.0
        num_train_batches = 0
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            q = query_tower(batch["query"])
            pos = doc_tower(batch["positive"])
            neg = doc_tower(batch["negative"])

            dst_pos = F.cosine_similarity(q, pos)
            dst_neg = F.cosine_similarity(q, neg)

            optimizer.zero_grad()
            loss = torch.max(zero, margin - dst_pos + dst_neg).mean()
            loss.backward()
            total_norm = 0
            for p in params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            optimizer.step()
            total_train_loss += loss.item()
            num_train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss = validate_model(
            query_tower, doc_tower, validation_dataloader, device
        )
        retrieval_acc = calculate_retrieval_metrics(
            query_tower, doc_tower, validation_dataset, device
        )

        logging.info(f"Epoch {epoch + 1}/{epochs}")
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Train Retrieval Accuracy: {retrieval_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best model checkpoint
            torch.save(
                {
                    "query_tower": query_tower.state_dict(),
                    "doc_tower": doc_tower.state_dict(),
                    "parameters": params,
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                    "dropout_rate": dropout_rate,
                },
                MODEL_FILE,
            )
    checkpoint = torch.load(MODEL_FILE)
    query_tower.load_state_dict(checkpoint["query_tower"])
    doc_tower.load_state_dict(checkpoint["doc_tower"])
    test_loss = validate_model(query_tower, doc_tower, test_dataloader, device)
    logging.info(f"Test Loss: {test_loss:.4f}")
    retrieval_acc = calculate_retrieval_metrics(
        query_tower, doc_tower, test_dataset, device
    )
    logging.info(f"Test Retrieval Accuracy: {retrieval_acc:.4f}")


if __name__ == "__main__":
    main()
