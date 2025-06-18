
from datasets import load_dataset
import pickle



def pull_ms_marco_dataset():
    """
    Pulls the MSMARCO dataset from Hugging Face and saves it as a Parquet file.
    """
    print("Pulling MSMARCO dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    dataset.to_parquet("ms_marco_train.parquet")
    print("MSMARCO dataset saved as 'msmarco_train.parquet'.")
#pull_ms_marco_dataset()

## Load, process the MS Marco dataset
def process_ms_marco_dataset():
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")

    container = []

    for data in dataset:
        passages = data['passages']
        pos = None
        neg = None

        for i in range(len(passages['is_selected'])):
            if passages['is_selected'][i] == 1 and pos is None:
                pos = passages['passage_text'][i]
            elif passages['is_selected'][i] == 0 and neg is None:
                neg = passages['passage_text'][i]
            if pos and neg:
                break

        container.append({
            "query": data['query'],
            "pos": pos,
            "neg": neg
        })

    return container

ms_marco = process_ms_marco_dataset()


with open("msmarco_triplets_untokenized.pkl", "wb") as f:
    pickle.dump(ms_marco, f)
