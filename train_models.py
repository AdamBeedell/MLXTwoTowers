import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
import numpy as np
from dataset import DATASET_FILE
from model import QueryTower, DocTower
from tokenizer import Word2VecTokenizer
from utils import MODEL_FILE

embed_dim = 300
margin = 0.5
query_learning_rate = 2e-5
doc_learning_rate = 1e-5
batch_size = 512
epochs = 15
dropout_rate = 0.1
patience = 3


class TripletDataLoader(DataLoader):
    def __init__(self, dataset, device):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )


def analyze_embedding_diversity(embeddings, name):
    """Check if embeddings are collapsing to similar values"""
    # Calculate pairwise similarities between embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    similarities = torch.mm(embeddings_norm, embeddings_norm.t())

    # Remove diagonal (self-similarities)
    similarities.fill_diagonal_(0)

    mean_sim = similarities.mean().item()
    std_sim = similarities.std().item()

    logging.info(
        f"{name} - Mean inter-embedding similarity: {mean_sim:.4f}, Std: {std_sim:.4f}"
    )
    return mean_sim, std_sim


def validate_model(query_tower, doc_tower, validation_dataloader, device):
    query_tower.eval()
    doc_tower.eval()

    total_loss = 0.0
    num_batches = 0
    zero = torch.tensor(0.0).to(device)

    all_margins = []
    pos_similarities = []
    neg_similarities = []

    correct_at_k = 0
    total_queries = 0

    with torch.no_grad():  # Important: no gradients during validation
        for i, batch in tqdm(enumerate(validation_dataloader), desc="Validation"):
            query, positive, negative = [b.to(device) for b in batch.values()]

            # Forward pass (same as training)
            q = query_tower(query)
            pos = doc_tower(positive)
            neg = doc_tower(negative)

            if i == 0:  # Only for first batch
                analyze_embedding_diversity(q, "Query embeddings")
                analyze_embedding_diversity(pos, "Positive doc embeddings")
                analyze_embedding_diversity(neg, "Negative doc embeddings")
                # Check if embeddings are too similar to each other
                q_mean = q.mean(dim=0)
                pos_mean = pos.mean(dim=0)
                neg_mean = neg.mean(dim=0)

                logging.info(
                    f"Embedding means - Query: {q_mean.norm():.4f}, Pos: {pos_mean.norm():.4f}, Neg: {neg_mean.norm():.4f}"
                )

                # Check variance across dimensions
                q_var = q.var(dim=0).mean()
                pos_var = pos.var(dim=0).mean()
                neg_var = neg.var(dim=0).mean()

                logging.info(
                    f"Embedding variance - Query: {q_var:.4f}, Pos: {pos_var:.4f}, Neg: {neg_var:.4f}"
                )
            # Calculate loss (same as training)
            dst_pos = F.cosine_similarity(q, pos)
            dst_neg = F.cosine_similarity(q, neg)

            pos_similarities.extend(dst_pos.cpu().numpy())
            neg_similarities.extend(dst_neg.cpu().numpy())
            margins = (dst_pos - dst_neg).cpu().numpy()
            all_margins.extend(margins)

            # Check if positive document ranks higher than negative
            correct_at_k += (dst_pos > dst_neg).sum().item()
            total_queries += dst_pos.size(0)

            loss = torch.max(zero, margin - dst_pos + dst_neg).mean()

            total_loss += loss.item()
            num_batches += 1

    logging.info(f"Validation Stats:")
    logging.info(
        f"  Pos Similarities - Mean: {np.mean(pos_similarities):.4f}, Std: {np.std(pos_similarities):.4f}"
    )
    logging.info(
        f"  Neg Similarities - Mean: {np.mean(neg_similarities):.4f}, Std: {np.std(neg_similarities):.4f}"
    )
    logging.info(
        f"  Margins - Mean: {np.mean(all_margins):.4f}, Std: {np.std(all_margins):.4f}"
    )
    logging.info(
        f"  Positive Margins: {(np.array(all_margins) > 0).sum()}/{len(all_margins)} ({(np.array(all_margins) > 0).mean()*100:.1f}%)"
    )
    recall_at_k = correct_at_k / total_queries if total_queries > 0 else 0.0

    query_tower.train()  # Set back to training mode
    doc_tower.train()

    return total_loss / num_batches, recall_at_k


def main():
    utils.setup_logging()
    device = utils.get_device()
    tokenizer = Word2VecTokenizer()
    vocab_size = tokenizer.vocab_size
    logging.info(f"Vocab size: {vocab_size}")
    query_tower = QueryTower(tokenizer.embeddings, embed_dim, dropout_rate).to(device)
    doc_tower = DocTower(tokenizer.embeddings, embed_dim, dropout_rate).to(device)
    logging.info(
        f"Query embeddings require grad: {query_tower.embedding.weight.requires_grad}"
    )
    logging.info(
        f"Doc embeddings require grad: {doc_tower.embedding.weight.requires_grad}"
    )
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
    params = [
        {"params": query_tower.parameters(), "lr": query_learning_rate},
        {"params": doc_tower.parameters(), "lr": doc_learning_rate},
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    logging.info(f"Starting training batch size {batch_size}")
    zero = torch.tensor(0.0)
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        total_train_loss = 0.0
        num_train_batches = 0
        all_train_margins = []
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            q = query_tower(batch["query"])
            pos = doc_tower(batch["positive"])
            neg = doc_tower(batch["negative"])

            dst_pos = F.cosine_similarity(q, pos)
            dst_neg = F.cosine_similarity(q, neg)

            margins = (dst_pos - dst_neg).detach().cpu()
            all_train_margins.extend(margins.tolist())

            all_params = []
            for group in optimizer.param_groups:
                all_params.extend(group["params"])
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            if i % 100 == 1:
                logging.info(f"Grad norm: {total_norm:.4f}")

            optimizer.zero_grad()
            loss = torch.max(zero, margin - dst_pos + dst_neg).mean()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            num_train_batches += 1

        logging.info(f"Epoch {epoch + 1}/{epochs}")
        margins_tensor = torch.tensor(all_train_margins)
        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss, retrieval_acc = validate_model(
            query_tower, doc_tower, validation_dataloader, device
        )
        scheduler.step(avg_val_loss)

        logging.info(f"Query learning rate: {optimizer.param_groups[0]["lr"]:.6f}")
        logging.info(f"Doc learning rate: {optimizer.param_groups[1]["lr"]:.6f}")
        logging.info(
            f"Train Margins - Avg: {margins_tensor.mean():.4f}, Min: {margins_tensor.min():.4f}, Max: {margins_tensor.max():.4f}"
        )
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Val Retrieval Accuracy: {retrieval_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
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
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    checkpoint = torch.load(MODEL_FILE)
    query_tower.load_state_dict(checkpoint["query_tower"])
    doc_tower.load_state_dict(checkpoint["doc_tower"])
    test_loss, retrieval_acc = validate_model(
        query_tower, doc_tower, test_dataloader, device
    )
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Retrieval Accuracy: {retrieval_acc:.4f}")


if __name__ == "__main__":
    main()
