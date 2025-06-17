### ABModel.py


### Import libraries

import torch
import torch.nn as nn
from datasets import load_dataset
import wandb
import bz2
import csv
import requests
from transformers import AutoTokenizer



###  define hyperparameters

min_freq = 10
context_size = 2
embed_dim = 300
epochs = 5
learning_rate = 0.01
batch_size = 8192


### Load datasets and/or weights

## download text8

def pull_wikipedia_data():
    downloadurl = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8"
    localfile = "wikipedia_data.txt.bz2"

    with requests.get(downloadurl, stream=True) as response:
        with bz2.open(localfile, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Data downloaded and saved to {localfile}")    


pull_wikipedia_data()


## Load the text8 dataset
def load_text8_dataset():
    """
    Loads the text8 dataset from a local file and returns it as a list of sentences.
    """
    with bz2.open("wikipedia_data.txt.bz2", 'rt') as f:
        text = f.read()
    sentences = text.split('\n')
    return sentences
text8_sentences = load_text8_dataset()



## download the MS Marco dataset
def pull_ms_marco_dataset():
    """
    Pulls the MSMARCO dataset from Hugging Face and saves it as a Parquet file.
    """
    print("Pulling MSMARCO dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    dataset.to_parquet("ms_marco_train.parquet")
    print("MSMARCO dataset saved as 'msmarco_train.parquet'.")
pull_ms_marco_dataset()

## Load the MS Marco dataset
def load_ms_marco_dataset():
    """
    Loads the MSMARCO dataset from a Parquet file and returns it as a list of passages.
    """
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    passages = [item['passages'] for item in dataset]
    return passages
ms_marco_passages = load_ms_marco_dataset()

## process the MS Marco dataset

###### Do stuff here to pull a query a passage where is_selected is True and a passage where is_selected is False
###### unsure if the urls or whatever are loaded in at this stage/at all
###### I expect that URLs are traditionally recalled, but that a title or doc ID or something is used to recall the passage, and that the passage is needed for context

## proess the text8 dataset
def process_text8_dataset(sentences):
    """
    Processes the text8 dataset by tokenizing sentences and removing short sentences.
    """
    processed_sentences = []
    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) > 5:  # Example threshold
            processed_sentences.append(tokens)
    return processed_sentences

processed_text8 = process_text8_dataset(text8_sentences)



### Make dataloader for text8 dataset
from torch.utils.data import DataLoader, Dataset
class Text8Dataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx], dtype=torch.long) 
    

### Make dataloader for MS Marco dataset
class MSMarcoDataset(Dataset):
    def __init__(self, passages):
        self.passages = passages

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        return torch.tensor(self.passages[idx], dtype=torch.long)  # Assuming passages are tokenized



### Make dataloader for triplet loss

class TripletDataLoader(DataLoader):
    def __init__(self, dataset, device):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )




### Define the ABModel class


class ABModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(ABModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x

    def get_embedding(self, x):
        return self.embedding(x)
    


### Initialize the model
def initialize_model(vocab_size, embed_dim):
    model = ABModel(vocab_size, embed_dim)
    return model

### Train the model

def train_model(model, dataloader, epochs, learning_rate):
    wandb.watch(model, log="all")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


####### Broken


# Init W&B run
run = wandb.init(
    project="MLXTwoTowers"
    entity="AdamBeedell-"  # org or personal
    config={                    # hyperparams here
        "min_freq" = min_freq,
        "context_size" = context_size,
        "embed_dim" = embed_dim,
        "epochs" = epochs,
        "learning_rate" = learning_rate,
        "vocab_size" = vocab_size, 
        "batch_size" = batch_size
    }
)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ### optimize for GPU if available
model.to(device)


dataloader = TripletDataLoader(dataloader, device)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"loss": loss.item()})
        wandb.log({"epoch": epoch + 1})  


    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}")



############# /broken




#### Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

### Upload the model to W&B
def upload_model_to_wandb(model, run):
    model_artifact = wb.Artifact("ABModel", type="model")
    model_artifact.add_file("ab_model.pth")
    run.log_artifact(model_artifact)
    print("Model uploaded to W&B") 


### recreate ABModel class to create 2nd tower

class ABModelTwoTowers(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(ABModelTwoTowers, self).__init__()
        self.query_embedding = nn.Embedding(vocab_size, embed_dim)
        self.doc_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, query, doc):
        query_emb = self.query_embedding(query)
        doc_emb = self.doc_embedding(doc)
        return query_emb, doc_emb

    def get_query_embedding(self, query):
        return self.query_embedding(query)

    def get_doc_embedding(self, doc):
        return self.doc_embedding(doc)
    
### Load previous weights into the two-tower model
def load_two_tower_weights(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Two-tower model weights loaded from {path}")
    return model

### Train the two-tower model
def train_two_tower_model(model, dataloader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            query, pos_doc, neg_doc = batch
            query = query.to(device)
            pos_doc = pos_doc.to(device)
            neg_doc = neg_doc.to(device)

            optimizer.zero_grad()
            query_emb, pos_doc_emb = model.get_query_embedding(query), model.get_doc_embedding(pos_doc)
            _, neg_doc_emb = model.get_doc_embedding(neg_doc)

            loss = criterion(query_emb, pos_doc_emb, neg_doc_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}")
        wandb.log({"loss": loss.item()})
        wandb.log({"epoch": epoch + 1})  

    return model

### Save the two-tower model
def save_two_tower_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Two-tower model saved to {path}")

### Upload the two-tower model to W&B
def upload_two_tower_model_to_wandb(model, run):
    model_artifact = wb.Artifact("ABModelTwoTowers", type="model")
    model_artifact.add_file("ab_model_two_towers.pth")
    run.log_artifact(model_artifact)
    print("Two-tower model uploaded to W&B")
