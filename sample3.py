import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random, time

print("\n\n\n\n\n\n\n\n\n\n\n\n")
print("#########################------- TTNN Sample V3 (Avg. Pooling only) ------################")
print("###################------- TTNN Sample V3 ------################")
print("###########------- TTNN Sample V3 ------################")
# --- Synthetic data generator (from earlier) ---

topics = {
    "countries": [("capital of {}", "{} is the capital city")],
    "programming": [("how to use {}", "guide to {}"), ("{} tutorial", "{} explained")],
    "animals": [("what does a {} eat", "{} diet and habitat")],
    "sports": [("{} player stats", "{} match history"), ("highlights of {}", "latest {} match")],
    "food": [("how to cook {}", "{} recipe"), ("health benefits of {}", "{} nutrition facts")],
    "history": [("when was {} discovered", "{} discovery year"), ("history of {}", "{} timeline")],
    "science": [("define {}", "{} definition"), ("explain {}", "{} in detail")],
    "music": [("top songs by {}", "{} discography"), ("best albums of {}", "{} music list")],
    "movies": [("review of {}", "{} movie review"), ("cast of {}", "{} film actors")],
}


entities = {
    "countries": ["paris", "berlin", "tokyo", "cairo", "london", "moscow", "madrid", "ottawa", "delhi", "beijing"],
    "programming": ["python", "javascript", "docker", "linux", "kubernetes", "flask", "django", "git"],
    "animals": ["lion", "tiger", "elephant", "penguin", "shark", "dolphin", "cat", "dog", "rabbit", "fox"],
    "sports": ["messi", "ronaldo", "federer", "jordan", "bolt", "hamilton", "serena", "lebron"],
    "food": ["pasta", "salmon", "chicken", "apple", "banana", "avocado", "carrot", "egg", "milk"],
    "history": ["internet", "steam engine", "printing press", "light bulb", "telephone"],
    "science": ["gravity", "relativity", "photosynthesis", "atom", "cell", "thermodynamics"],
    "music": ["beatles", "beyonce", "eminem", "mozart", "adele", "coldplay", "rihanna"],
    "movies": ["inception", "avatar", "titanic", "interstellar", "joker", "matrix", "gladiator"],
}

### 1. Synthetic Pairs
#
###

def generate_synthetic_pairs(n_per_topic=150):
    pairs = []
    for topic, templates in topics.items():
        for _ in range(n_per_topic):
            entity = random.choice(entities[topic])
            template = random.choice(templates)
            query = template[0].format(entity)
            doc = template[1].format(entity)
            pairs.append((query.lower(), doc.lower()))
    return pairs

pairs = generate_synthetic_pairs(n_per_topic=150)
print("1- synthetic pairs created")
time.sleep(2)
#print(pairs[-10:])

### 2. Build vocabulary ---
#
###
all_text = [w for pair in pairs for text in pair for w in text.split()]
vocab = {w: i+1 for i, w in enumerate(set(all_text))}  # +1 for padding idx=0
vocab_size = len(vocab) + 1  # plus padding
print("2- Vocab created")
time.sleep(2)

def text_to_indices(text):
    return [vocab.get(w, 0) for w in text.split()]  # 0 if OOV

### 3. Dataset ---
#
###
class TwoTowerDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, doc = self.pairs[idx]
        return torch.tensor(text_to_indices(query)), torch.tensor(text_to_indices(doc))

# --- Collate for padding ---
def collate_batch(batch):
    queries, docs = zip(*batch)
    queries_pad = nn.utils.rnn.pad_sequence(queries, batch_first=True)
    docs_pad = nn.utils.rnn.pad_sequence(docs, batch_first=True)
    return queries_pad, docs_pad

dataset = TwoTowerDataset(pairs)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
time.sleep(2)
print(f"Dataset size: {len(dataset)}")

print("3- Two tower dataset?, padding and dataloader")
time.sleep(2)


### 4. Two-Tower Model ---
#
###

class TwoTowerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.query_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.doc_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Instead of LSTM, just a linear projection after average pooling
        self.query_fc = nn.Linear(embedding_dim, 64)
        self.doc_fc = nn.Linear(embedding_dim, 64)

    def forward(self, query_idxs, doc_idxs):
        q_emb = self.query_embedding(query_idxs)  # (batch, seq_len, embed_dim)
        d_emb = self.doc_embedding(doc_idxs)      # (batch, seq_len, embed_dim)
        
        # Average pooling across sequence length, ignoring padding (assumes padding idx=0)
        q_mask = (query_idxs != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
        d_mask = (doc_idxs != 0).unsqueeze(-1).float()
        
        q_sum = (q_emb * q_mask).sum(dim=1)               # sum embeddings over seq_len
        q_len = q_mask.sum(dim=1).clamp(min=1e-9)         # count valid tokens
        q_avg = q_sum / q_len                              # average embedding
        
        d_sum = (d_emb * d_mask).sum(dim=1)
        d_len = d_mask.sum(dim=1).clamp(min=1e-9)
        d_avg = d_sum / d_len
        
        # Linear projection + normalize
        q_out = self.query_fc(q_avg)
        d_out = self.doc_fc(d_avg)
        
        q_norm = F.normalize(q_out, dim=1)
        d_norm = F.normalize(d_out, dim=1)
        
        return q_norm, d_norm




print("4- TwoTower Model Defined")
time.sleep(2)


### 5. Loss Function Defined
#
###
def triplet_loss_batch(q_emb, d_emb, margin=0.5):
    """
    q_emb: (batch_size, dim)
    d_emb: (batch_size, dim)
    """
    if q_emb.dim() != 2 or d_emb.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got q_emb: {q_emb.shape}, d_emb: {d_emb.shape}")

    sim_matrix = torch.matmul(q_emb, d_emb.T)  # shape: (B, B)
    positive_scores = sim_matrix.diag()        # (B,)

    batch_size = q_emb.size(0)
    loss = 0

    for i in range(batch_size):
        pos = positive_scores[i]
        # All negatives for the i-th query
        neg = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        triplet_losses = F.relu(margin + neg - pos)
        loss += triplet_losses.mean()

    return loss / batch_size

print("5- Loss Function Created")
time.sleep(2)

### 6. Training TTNN
#
###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for q_batch, d_batch in dataloader:
        q_batch, d_batch = q_batch.to(device), d_batch.to(device)
        optimizer.zero_grad()
        q_emb, d_emb = model(q_batch, d_batch)

        
        # DEBUG: Print shapes here
	#print("q_emb shape:", q_emb.shape)  # Expect: (batch_size, embedding_dim)
	#print("d_emb shape:", d_emb.shape)  # Expect: (batch_size, embedding_dim)
        

        #loss = contrastive_loss(q_emb, d_emb)
        loss = triplet_loss_batch(q_emb, d_emb)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}", end=' | ')



print("6- TwoTowerModel Trained")
time.sleep(2)


print("###########------- Everything after here is Inference ------################")
time.sleep(2)
### Inference After Here
#
###

model.eval()
test_query = "capital of paris"
test_doc_1 = "paris is the capital of france"
test_doc_2 = "how to cook pasta"

#def embed_query(text):
#    indices = torch.tensor(text_to_indices(text)).unsqueeze(0).to(device)
#    with torch.no_grad():
#        q_emb = model.query_fc(model.query_lstm(model.query_embedding(indices))[1][0].squeeze(0))
#        q_emb = F.normalize(q_emb, dim=1)
#    return q_emb[0].cpu()
#
#def embed_doc(text):
#    indices = torch.tensor(text_to_indices(text)).unsqueeze(0).to(device)
#    with torch.no_grad():
#        d_emb = model.doc_fc(model.doc_lstm(model.doc_embedding(indices))[1][0].squeeze(0))
#        d_emb = F.normalize(d_emb, dim=1)
#    return d_emb[0].cpu()


def embed_query(text):
    indices = torch.tensor(text_to_indices(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        q_embs = model.query_embedding(indices)                 # (1, seq_len, embed_dim)
        q_mask = (indices != 0).unsqueeze(-1).float()           # (1, seq_len, 1)
        q_avg = (q_embs * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1e-9)
        q_out = model.query_fc(q_avg)
        q_norm = F.normalize(q_out, dim=1)
    return q_norm[0].cpu()


def embed_doc(text):
    indices = torch.tensor(text_to_indices(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        d_embs = model.doc_embedding(indices)                   # (1, seq_len, embed_dim)
        d_mask = (indices != 0).unsqueeze(-1).float()           # (1, seq_len, 1)
        d_avg = (d_embs * d_mask).sum(dim=1) / d_mask.sum(dim=1).clamp(min=1e-9)
        d_out = model.doc_fc(d_avg)
        d_norm = F.normalize(d_out, dim=1)
    return d_norm[0].cpu()




def similarity_score(q, d):
    return torch.dot(q, d).item()


queries = [
    "capital of paris",
    "how to use docker",
    "what does a lion eat",]

documents = [
    "paris is the capital city",
    "guide to docker containers",
    "lion diet and habitat",
    "latest football match highlights",]

for query in queries[:1]:
    q_emb = embed_query(query)
    print(f"\nQuery: '{query}'")
    for doc in documents:
        d_emb = embed_doc(doc)
        sim = similarity_score(q_emb, d_emb)
        print(f"  Doc: '{doc}' -> similarity: {sim:.4f}")


print("Loss is {:.4f}".format(loss))
