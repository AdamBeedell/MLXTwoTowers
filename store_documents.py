import logging

import redis
import numpy as np
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from datasets import load_dataset
from tqdm import tqdm
import torch

import utils
from model import Tower


def load_document_corpus():
    """Load all unique documents from MS MARCO dataset"""
    logging.info("Loading MS MARCO document corpus...")

    # Load the dataset
    ms_marco_data = load_dataset("ms_marco", "v1.1")

    # Extract all unique documents from all splits
    all_documents = {}  # Use dict to avoid duplicates

    for split_name in ["train", "validation", "test"]:
        logging.info(f"Processing {split_name} split...")
        split_data = ms_marco_data[split_name]

        for row in tqdm(split_data, desc=f"Extracting docs from {split_name}"):
            passages = row["passages"]
            passage_texts = passages["passage_text"]

            # Each passage gets a unique ID
            for i, passage_text in enumerate(passage_texts):
                # Create unique document ID
                doc_id = f"{split_name}_{row.get('query_id', len(all_documents))}_{i}"

                # Store document
                all_documents[doc_id] = {
                    "id": doc_id,
                    "text": passage_text,
                    "split": split_name,
                    "query_id": row.get("query_id"),
                    "is_selected": (
                        passages["is_selected"][i]
                        if "is_selected" in passages
                        else False
                    ),
                }

    # Convert to list
    documents = list(all_documents.values())
    logging.info(f"Loaded {len(documents)} unique documents from MS MARCO")

    return documents


def encode_all_documents(tokenizer, doc_tower, documents, device, batch_size=1000):
    """Encode all documents and return embeddings with IDs"""
    doc_tower.eval()

    all_embeddings = []
    doc_metadata = []

    with torch.no_grad():
        for i in tqdm(range(0, len(documents), batch_size), desc="Encoding documents"):
            batch_docs = documents[i : i + batch_size]
            batch_texts = [doc["text"] for doc in batch_docs]

            # Tokenize batch
            tokenized = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            )["input_ids"].to(device)

            # Encode
            embeddings = doc_tower(tokenized)

            all_embeddings.extend(embeddings.cpu().numpy())
            # Store metadata for each document
            for doc in batch_docs:
                doc_metadata.append(
                    {
                        "id": doc["id"],
                        "text": (
                            doc["text"][:200] + "..."
                            if len(doc["text"]) > 200
                            else doc["text"]
                        ),
                        # Truncate for storage
                        "split": doc["split"],
                        "query_id": doc.get("query_id"),
                        "is_selected": doc.get("is_selected", False),
                    }
                )
    logging.info(
        f"Saving {len(doc_metadata)} metadata and {len(all_embeddings)} embeddings to Redis"
    )
    return doc_metadata, np.array(all_embeddings)


def store_embeddings_in_redis(doc_metadata, embeddings, redis_client):
    """Store document embeddings in Redis"""

    redis_client.flushdb()
    pipeline = redis_client.pipeline(transaction=False)
    for doc_meta, embedding in tqdm(
        zip(doc_metadata, embeddings), desc="Storing in Redis"
    ):
        # Store embedding as binary data
        embedding_bytes = embedding.astype(np.float32).tobytes()
        doc_id = doc_meta["id"]
        pipeline.hset(
            f"doc:{doc_id}",
            mapping={"text": doc_meta["text"], "embedding": embedding_bytes},
        )

        if len(pipeline.command_stack) >= 1000:
            pipeline.execute()
    pipeline.execute()

    logging.info(f"Stored {len(doc_metadata)} document embeddings in Redis")


def create_redis_index(redis_client, dim):

    try:
        redis_client.ft("doc_index").dropindex(delete_documents=False)
    except Exception:
        pass

    logging.info("Creating Redis index...")
    redis_client.ft("doc_index").create_index(
        fields=[
            TextField("text"),
            VectorField(
                "embedding",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
            ),
        ],
        definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH),
    )


def main():
    utils.setup_logging()
    device = utils.get_device()
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    checkpoint = torch.load(utils.MODEL_FILE, map_location=device)
    doc_tower = Tower(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    doc_tower.load_state_dict(checkpoint["doc_tower"])

    logging.info("Loading MS MARCO dataset...")
    documents = load_document_corpus()

    logging.info("Encoding documents...")
    tokenizer = utils.get_tokenizer()
    doc_metadata, embeddings = encode_all_documents(
        tokenizer, doc_tower, documents, device
    )

    logging.info("Storing in Redis...")
    store_embeddings_in_redis(doc_metadata, embeddings, redis_client)

    create_redis_index(redis_client, embeddings.shape[1])

    # Store metadata
    redis_client.set("embedding_dim", embeddings.shape[1])
    redis_client.set("total_docs", len(doc_metadata))

    logging.info("Document cache setup complete!")


if __name__ == "__main__":
    main()
