import json
import logging
from contextlib import asynccontextmanager
import os
import numpy as np
import redis
import torch
from fastapi import FastAPI
from model import Tower
import utils
from redis.commands.search.query import Query

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
    utils.setup_logging()
    device = utils.get_device()
    checkpoint = torch.load(utils.MODEL_FILE, map_location=device)

    query_tower = Tower(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    query_tower.load_state_dict(checkpoint["query_tower"])
    tokenizer = utils.get_embeddings()
    redis_client = redis.Redis(host="redis-stack", port=6379, db=0)
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
    query_tower.eval()
    with torch.no_grad():
        tokenized_query = tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(device)

        query_embedding = (
            query_tower(tokenized_query).cpu().numpy().astype(np.float32).tobytes()
        )

    redis_query = f"*=>[KNN {top_k} @embedding $vec_param]"
    query_obj = (
        Query(redis_query)
        .return_fields("text", "embedding", "text")
        .with_scores()
        .paging(0, top_k)
        .dialect(2)
    )

    logging.info(
        "Query embedding norm:",
        np.linalg.norm(np.frombuffer(query_embedding, dtype=np.float32)),
    )
    results = redis_client.ft("doc_index").search(
        query_obj, query_params={"vec_param": query_embedding}
    )

    return [(doc.id, float(doc.score), doc.text) for doc in results.docs]


##### Log The Request #####
def log_request(log_path, message):
    # print the message and then write it to the log
    print(message)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


##### Read The Logs #####
def read_logs(log_path):
    # read the logs from the log_path
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as log_file:
        lines = log_file.readlines()
    return [line.strip() for line in lines]
