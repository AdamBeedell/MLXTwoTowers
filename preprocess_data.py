import logging
from datasets import load_dataset
import torch
from dataset import TripletDataset, DATASET_FILE
import utils
from tokenizer import Word2VecTokenizer


def main():
    utils.setup_logging()
    device = utils.get_device()
    tokenizer = Word2VecTokenizer()

    logging.info("Loading MS MARCO dataset...")
    ms_marco_data = load_dataset("ms_marco", "v1.1")

    logging.info("Creating training dataset...")
    train_dataset = TripletDataset(ms_marco_data["train"], tokenizer, device)

    logging.info("Creating validation dataset...")
    validation_dataset = TripletDataset(ms_marco_data["validation"], tokenizer, device)

    logging.info("Creating test dataset...")
    test_dataset = TripletDataset(ms_marco_data["test"], tokenizer, device)

    logging.info("Saving preprocessed datasets...")
    torch.save(
        {
            "train_triplets": train_dataset.triplets,
            "val_triplets": validation_dataset.triplets,
            "test_triplets": test_dataset.triplets,
            "tokenizer": tokenizer,
        },
        DATASET_FILE,
    )

    logging.info(
        f"Saved datasets with sizes: train={len(train_dataset)}, val={len(validation_dataset)}, test={len(test_dataset)}"
    )


if __name__ == "__main__":
    main()
