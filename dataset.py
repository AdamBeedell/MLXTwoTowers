import logging
import random

import numpy as np
from torch.utils.data import Dataset

DATASET_FILE = "data/datasets.pt"


class TripletDataset(Dataset):
    def __init__(self, ms_marco_data, tokenizer, device):
        self.triplets = []
        self.tokenizer = tokenizer
        self.device = device

        # Group by query to create triplets
        query_groups = {}
        for row in ms_marco_data:
            query = row["query"]
            passages = row["passages"]
            passage_texts = passages["passage_text"]
            is_selected = np.array(passages["is_selected"])
            nonzero_indices = np.nonzero(is_selected)[0]
            if len(nonzero_indices) == 0:
                continue  # there's no positive passage, skip this query
            selected_item = nonzero_indices[0]
            valid_indices = [i for i in range(len(is_selected)) if i != selected_item]
            if len(valid_indices) == 0:
                continue  # there's no negative passage, skip this query
            negative_item = random.choice(valid_indices)
            if selected_item == negative_item:
                logging.error(f"Selected and negative items are the same: {row}")
                raise Exception("Selected and negative items are the same")
            if query not in query_groups:
                query_groups[query] = {}
            query_groups[query]["pos"] = passage_texts[selected_item]
            query_groups[query]["neg"] = passage_texts[negative_item]

        # Create (query, positive_passage, negative_passage) triplets
        for query, pair in query_groups.items():
            if len(pair["pos"]) > 0 and len(pair["neg"]) > 0:
                self.triplets.append(
                    {
                        "query": self.tokenize(query),
                        "positive": self.tokenize(pair["pos"]),
                        "negative": self.tokenize(pair["neg"]),
                    }
                )

    def tokenize(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].squeeze(0)

    def __getitem__(self, idx):
        return self.triplets[idx]

    def __len__(self):
        return len(self.triplets)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
