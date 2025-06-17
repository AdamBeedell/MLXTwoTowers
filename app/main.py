import json
from contextlib import asynccontextmanager
import os
import pickle
import numpy as np
import redis
import torch
from fastapi import FastAPI
from transformers import AutoTokenizer
from model import Tower
import utils

# Write the model version here or find some way to derive it from the model
# eg. from the model files name
model_version = "0.1.0"

# Set the log path.
# This should be a directory that is writable by the application.
# In a docker container, you can use /var/log/ as the directory.
# Mount this directory to a volume on the host machine to persist the logs.
log_dir_path = "/var/log/app"
log_path = f"{log_dir_path}/V-{model_version}.log"


query_tower = None
tokenizer = None
device = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_tower, tokenizer, device, redis_client

    device = utils.get_device()
    checkpoint = torch.load(utils.outfile, map_location=device)

    query_tower = Tower(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    query_tower.load_state_dict(checkpoint["query_tower"])
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    yield


app = FastAPI(lifespan=lifespan)


# Define the endpoints
@app.get("/ping")
def ping():
    return "ok"


@app.get("/version")
def version():
    return {"version": model_version}


@app.get("/logs")
def logs():
    return read_logs(log_path)


@app.get("/search")
def search(query):
    global query_tower, redis_client, tokenizer, device
    if (query_tower is None) or (tokenizer is None) or (redis_client is None):
        raise Exception("App not initialized.")
    start_time = os.times().elapsed  # Start time for latency calculation
    documents = do_search(
        query, query_tower, redis_client, tokenizer, device
    )  # Placeholder for actual search
    end_time = os.times().elapsed  # End time for latency calculation
    latency = (end_time - start_time) * 1000

    message = {
        "Latency": latency,
        "Version": model_version,
        "Timestamp": end_time,
        "Input": query,
        "Documents": documents,
    }

    log_request(log_path, json.dumps(message))
    return {"documents": documents}


def do_search(query, query_tower, redis_client, tokenizer, device, top_k=5):
    """Search MS MARCO documents with metadata"""
    query_tower.eval()

    # Encode query
    with torch.no_grad():
        tokenized_query = tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(device)

        query_embedding = query_tower(tokenized_query).cpu().numpy()[0]

    # Get all document embeddings
    similarities = []

    for doc_id_bytes, embedding_bytes in redis_client.hscan_iter("doc_embeddings"):
        doc_id = doc_id_bytes.decode("utf-8")
        embedding = pickle.loads(embedding_bytes)

        # Calculate cosine similarity
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities.append((doc_id, float(similarity)))

    # Get top-k
    top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    # Fetch metadata for top results
    results = []
    for doc_id, similarity in top_docs:
        metadata_bytes = redis_client.hget("doc_metadata", doc_id)
        metadata = pickle.loads(metadata_bytes)
        results.append((doc_id, similarity, metadata["text"]))

    return results


##### Log The Request #####
def log_request(log_path, message):
    # print the message and then write it to the log
    pass


##### Read The Logs #####
def read_logs(log_path):
    # read the logs from the log_path
    pass
