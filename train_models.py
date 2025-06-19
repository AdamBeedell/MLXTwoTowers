import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import autocast

import wandb
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
import numpy as np
from dataset import DATASET_FILE
from model import QueryTower, DocTower
from tokenizer import Word2VecTokenizer
from utils import MODEL_FILE

hyperparameters = {
    "embed_dim": 300,
    "margin": 0.3,
    "query_learning_rate": 2e-5,
    "doc_learning_rate": 1e-5,
    "batch_size": 256,
    "epochs": 50,
    "dropout_rate": 0.1,
    "patience": 5,
}


class TripletDataLoader(DataLoader):
    def __init__(self, dataset, device):
        num_workers = 8 if device.type == "cuda" else 0 if device.type == "mps" else 4
        super().__init__(
            dataset,
            batch_size=hyperparameters["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )


def analyze_embedding_diversity(run, embeddings, name, epoch):
    """Check if embeddings are collapsing to similar values"""
    # Calculate pairwise similarities between embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    similarities = torch.mm(embeddings_norm, embeddings_norm.t())

    # Remove diagonal (self-similarities)
    similarities.fill_diagonal_(0)

    mean_sim = similarities.mean().item()
    std_sim = similarities.std().item()

    run.log(
        {
            f"{name.lower().replace(' ', '_')}_mean_inter_embedding_similarity": mean_sim,
            f"{name.lower().replace(' ', '_')}_std_inter_embedding_similarity": std_sim,
        },
        step=epoch,
    )
    return mean_sim, std_sim


def validate_model(run, query_tower, doc_tower, validation_dataloader, epoch, device):
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
                analyze_embedding_diversity(run, q, "query", epoch)
                analyze_embedding_diversity(run, pos, "positive doc", epoch)
                analyze_embedding_diversity(run, neg, "negative doc", epoch)

                # Check if embeddings are too similar to each other
                q_mean = q.mean(dim=0)
                pos_mean = pos.mean(dim=0)
                neg_mean = neg.mean(dim=0)

                # Check variance across dimensions
                q_var = q.var(dim=0).mean()
                pos_var = pos.var(dim=0).mean()
                neg_var = neg.var(dim=0).mean()

                run.log(
                    {
                        "embedding_means_query_norm": q_mean.norm().item(),
                        "embedding_means_pos_norm": pos_mean.norm().item(),
                        "embedding_means_neg_norm": neg_mean.norm().item(),
                        "embedding_variance_query": q_var.item(),
                        "embedding_variance_pos": pos_var.item(),
                        "embedding_variance_neg": neg_var.item(),
                    },
                    step=epoch,
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

            loss = torch.max(zero, hyperparameters["margin"] - dst_pos + dst_neg).mean()

            total_loss += loss.item()
            num_batches += 1

    run.log(
        {
            "validation_pos_similarities_mean": np.mean(pos_similarities),
            "validation_pos_similarities_std": np.std(pos_similarities),
            "validation_neg_similarities_mean": np.mean(neg_similarities),
            "validation_neg_similarities_std": np.std(neg_similarities),
            "validation_margins_mean": np.mean(all_margins),
            "validation_margins_std": np.std(all_margins),
            "validation_positive_margins_count": (np.array(all_margins) > 0).sum(),
            "validation_positive_margins_ratio": (np.array(all_margins) > 0).mean()
            * 100,
        },
        step=epoch,
    )
    recall_at_k = correct_at_k / total_queries if total_queries > 0 else 0.0

    query_tower.train()  # Set back to training mode
    doc_tower.train()

    return total_loss / num_batches, recall_at_k


def training_loop_core(
    batch, query_tower, doc_tower, all_train_margins, optimizer, zero
):
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

    loss = torch.max(zero, hyperparameters["margin"] - dst_pos + dst_neg).mean()
    return loss, total_norm


def main():
    utils.setup_logging()
    device = utils.get_device()

    tokenizer = Word2VecTokenizer()
    vocab_size = tokenizer.vocab_size

    query_tower = QueryTower(
        tokenizer.embeddings,
        hyperparameters["embed_dim"],
        hyperparameters["dropout_rate"],
    ).to(device)
    doc_tower = DocTower(
        tokenizer.embeddings,
        hyperparameters["embed_dim"],
        hyperparameters["dropout_rate"],
    ).to(device)

    config = {
        **hyperparameters,
        "vocab_size": vocab_size,
        "query_trained": query_tower.embedding.weight.requires_grad,
        "doc_trained": doc_tower.embedding.weight.requires_grad,
    }
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="mlx-institute",
        # Set the wandb project where this run will be logged.
        project="TwoTowers",
        # Track hyperparameters and run metadata.
        config=config,
    )

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

    scaler = torch.amp.GradScaler(device=device.type)

    params = [
        {
            "params": query_tower.parameters(),
            "lr": hyperparameters["query_learning_rate"],
        },
        {"params": doc_tower.parameters(), "lr": hyperparameters["doc_learning_rate"]},
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    zero = torch.tensor(0.0).to(device)
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0
    for epoch in range(hyperparameters["epochs"]):
        total_train_loss = 0.0
        num_train_batches = 0
        all_train_margins = []
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            if device.type == "mps":
                loss, total_norm = training_loop_core(
                    batch, query_tower, doc_tower, all_train_margins, optimizer, zero
                )
                loss.backward()
                optimizer.step()
            else:
                with autocast(device_type=device.type):
                    loss, total_norm = training_loop_core(
                        batch,
                        query_tower,
                        doc_tower,
                        all_train_margins,
                        optimizer,
                        zero,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_train_loss += loss.item()
            num_train_batches += 1

        logging.info(f"Epoch {epoch + 1}/{hyperparameters["epochs"]}")
        margins_tensor = torch.tensor(all_train_margins)
        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss, retrieval_acc = validate_model(
            run, query_tower, doc_tower, validation_dataloader, epoch, device
        )
        scheduler.step(avg_val_loss)

        run.log(
            {
                "query_learning_rate": optimizer.param_groups[0]["lr"],
                "doc_learning_rate": optimizer.param_groups[1]["lr"],
                "train_margins_avg": margins_tensor.mean().item(),
                "train_margins_min": margins_tensor.min().item(),
                "train_margins_max": margins_tensor.max().item(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_retrieval_accuracy": retrieval_acc,
                "grad_norm": total_norm.item(),
            },
            step=epoch,
        )
        last_epoch += 1
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "query_tower": query_tower.state_dict(),
                    "doc_tower": doc_tower.state_dict(),
                    "parameters": params,
                    "vocab_size": vocab_size,
                    "embed_dim": hyperparameters["embed_dim"],
                    "dropout_rate": hyperparameters["dropout_rate"],
                },
                MODEL_FILE,
            )
        else:
            patience_counter += 1
            if patience_counter >= hyperparameters["patience"]:
                run.log({"early_stopping_epochs": epoch + 1})
                break
    checkpoint = torch.load(MODEL_FILE)
    query_tower.load_state_dict(checkpoint["query_tower"])
    doc_tower.load_state_dict(checkpoint["doc_tower"])
    test_loss, retrieval_acc = validate_model(
        run,
        query_tower,
        doc_tower,
        test_dataloader,
        last_epoch + 1,
        device,
    )
    run.log(
        {"test_loss": test_loss, "test_retrieval_accuracy": retrieval_acc},
        step=last_epoch + 1,
    )
    artifact = wandb.Artifact(name="two_tower_model", type="model")
    artifact.add_file(MODEL_FILE)
    run.log_artifact(artifact)
    run.finish(0)


if __name__ == "__main__":
    main()
