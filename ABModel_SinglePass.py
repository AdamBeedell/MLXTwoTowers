### ABModel_SinglePass.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

# Load tokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)


### Hyperparameters
min_freq = 10
max_length = 500
embed_dim = 300
epochs = 20
learning_rate = 0.01
batch_size = 128
vocab_size = 30522

# Load tokenized MSMARCO triplets
def load_ms_marco_dataset_tokenized_idx():
    return pickle.load(open("msmarco_triplets_tokenized_to_idx.pkl", "rb"))

# Dataset + Dataloader
class TripletDataLoader2(DataLoader):
    def __init__(self, dataset, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        queries = [x["query_input_ids"] for x in batch]
        positives = [x["pos_input_ids"] for x in batch]
        negatives = [x["neg_input_ids"] for x in batch]

        return {
            "query": torch.nn.utils.rnn.pad_sequence([torch.tensor(q) for q in queries], batch_first=True, padding_value=0),
            "pos": torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in positives], batch_first=True, padding_value=0),
            "neg": torch.nn.utils.rnn.pad_sequence([torch.tensor(n) for n in negatives], batch_first=True, padding_value=0),
        }

# Two-tower embedding model
class ABModelTwoTowers(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.query_embedding = nn.Embedding(vocab_size, embed_dim)
        self.doc_embedding = nn.Embedding(vocab_size, embed_dim)

    def get_query_embedding(self, query):
        return self.query_embedding(query)

    def get_doc_embedding(self, doc):
        return self.doc_embedding(doc)

# Training function
def train_two_tower_model(model, dataloader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            query = batch["query"].to(device)
            pos_doc = batch["pos"].to(device)
            neg_doc = batch["neg"].to(device)

            optimizer.zero_grad()
            query_emb = model.get_query_embedding(query).mean(dim=1)    ###### this sucks change this
            pos_emb = model.get_doc_embedding(pos_doc).mean(dim=1)
            neg_emb = model.get_doc_embedding(neg_doc).mean(dim=1)

            loss = criterion(query_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
        wandb.log({"loss": total_loss / len(dataloader), "epoch": epoch + 1})

    return model

# Save model
def save_two_tower_model(model, path="ab_model_two_towers.pth"):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

# Upload model to W&B
def upload_two_tower_model_to_wandb(model, run, path="ab_model_two_towers.pth"):
    artifact = wandb.Artifact("ABModelTwoTowers", type="model")
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Uploaded model to W&B")

# Run training
if __name__ == "__main__":
    ms_marco = load_ms_marco_dataset_tokenized_idx()
    dataset = ms_marco
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = TripletDataLoader2(dataset, tokenizer, max_length, device)

    run = wandb.init(
        project="MLXTwoTowers",
        entity="AdamBeedell-",
        config={
            "min_freq": min_freq,
            "max_length": max_length,
            "embed_dim": embed_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "vocab_size": vocab_size,
            "batch_size": batch_size
        }
    )

    model = ABModelTwoTowers(vocab_size, embed_dim)
    model = train_two_tower_model(model, dataloader, epochs, learning_rate)
    save_two_tower_model(model)
    upload_two_tower_model_to_wandb(model, run)


    wandb.finish()


def infer(model, query_input_ids, doc_input_ids, device):
    model.eval()
    with torch.no_grad():
        query_tensor = torch.tensor(query_input_ids).unsqueeze(0).to(device)  # [1, seq_len]
        doc_tensor = torch.tensor(doc_input_ids).unsqueeze(0).to(device)      # [1, seq_len]

        query_emb = model.get_query_embedding(query_tensor)  # [1, seq_len, embed_dim]
        doc_emb = model.get_doc_embedding(doc_tensor)        # [1, seq_len, embed_dim]

        # Average pooling over sequence length
        query_vec = query_emb.mean(dim=1)  # [1, embed_dim]
        doc_vec = doc_emb.mean(dim=1)      # [1, embed_dim]

        score = F.cosine_similarity(query_vec, doc_vec)  # [1]

    return score.item()

#### define query_ids, doc_ids

#### here's where precomputing all the tokens fails to pay off

score = infer(model, query_ids, doc_ids, device)
print("Similarity:", score)