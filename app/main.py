import json
import logging
import struct
from contextlib import asynccontextmanager
import os
import numpy as np
import redis
import torch
from fastapi import FastAPI
from model import QueryTower
import utils
from redis.commands.search.query import Query

from tokenizer import Word2VecTokenizer

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
    tokenizer = Word2VecTokenizer()
    checkpoint = torch.load(utils.MODEL_FILE, map_location=device)

    query_tower = QueryTower(
        tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    query_tower.load_state_dict(checkpoint["query_tower"])
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
        "Latency": int(latency),
        "Version": model_version,
        "Input": query,
        "Document": documents[0][:50],
    }

    log_request(log_path, json.dumps(message))
    return {"documents": documents}


def do_search(query, query_tower, redis_client, tokenizer, device, top_k=5):
    query_tower.eval()
    with torch.no_grad():
        tokenized_query = tokenizer(query)["input_ids"].to(device)

        query_embedding = (
            query_tower(tokenized_query).cpu().numpy().astype(np.float32).flatten()
        )
        logging.info(
            f"Query embedding norm: {np.linalg.norm(query_embedding):.8f}",
        )
        # Ensure the vector is normalized (Redis cosine similarity expects this)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Convert to bytes for Redis
        query_embedding_bytes = struct.pack(
            f"<{len(query_embedding)}f", *query_embedding
        )

        # --- START DEBUG CODE ---
        # Save the numpy array to a file for external verification
        np.save("debug_query_vector.npy", query_embedding)
        logging.info("DEBUG: Saved query vector to debug_query_vector.npy")
        # --- END DEBUG CODE ---

    redis_query = f"*=>[KNN {top_k} @embedding $vec_param AS vector_score]"
    query_obj = (
        Query(redis_query)
        .return_fields("text", "vector_score")
        .sort_by("vector_score")
        .paging(0, top_k)
        .dialect(2)
    )

    results = redis_client.ft("doc_index").search(
        query_obj, query_params={"vec_param": query_embedding_bytes}
    )

    # Return results with proper similarity scores
    search_results = []
    for doc in results.docs:
        doc_id = doc.id
        distance = float(doc.vector_score)
        similarity = 1.0 - distance  # Convert distance to similarity
        text = doc.text
        search_results.append((doc_id, similarity, text))

    logging.info(f"Found {len(search_results)} results:")
    for i, (doc_id, similarity, text) in enumerate(search_results):
        logging.info(f"  {i + 1}. {doc_id} (similarity: {similarity:.6f}): {text}...")

    # --- START: FINAL DEBUG BLOCK ---
    logging.info("--- RUNNING MANUAL VERIFICATION WITHIN APP ---")
    # Manually check similarity against a known-good document from test_found_documents.py
    known_good_doc_ids = ["doc:test_0_2", results.docs[0].id]
    for known_good_doc_id in known_good_doc_ids:
        good_doc_bytes = redis_client.hget(known_good_doc_id, "embedding")

        if good_doc_bytes:
            good_doc_vec = np.frombuffer(good_doc_bytes, dtype=np.float32)

            # 'query_embedding' is the normalized numpy array from earlier in this function
            # We use .flatten() just to be 100% sure it's 1D for the dot product
            manual_similarity = np.dot(query_embedding.flatten(), good_doc_vec)

            logging.info(
                f"MANUAL SIMILARITY CHECK with '{known_good_doc_id}': {manual_similarity:.6f}"
            )
        else:
            logging.info(
                f"Could not find known-good doc '{known_good_doc_id}' for manual check."
            )
        # --- END: FINAL DEBUG BLOCK ---

    return search_results


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
