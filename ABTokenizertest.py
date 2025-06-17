from transformers import BertTokenizer
import bz2


def load_text8_dataset():
    """
    Loads the text8 dataset from a local file.
    """
    with bz2.open("wikipedia_data.txt.bz2", 'rt') as f:
        text = f.read()
    return text


text8 = load_text8_dataset()  ##### Load the text8 dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)  ##### load a tokenizer which doesnt care about length of text

# 1. Get all special token *names* (standard + extra)
special_token_names = tokenizer.special_tokens_map_extended.keys()

# 2. Print tokens and their string values
for name in special_token_names:
    print(f"{name}: {getattr(tokenizer, name)}")

tokens = tokenizer.tokenize(text8)
print(tokens[:100]) 

tokenizer.vocab_size  # Check the vocabulary size


def split_into_chunks(text, chunk_size=512):
    """
    Splits the text into chunks of a specified size.
    
    Args:
        text (str): The input text to be split.
        chunk_size (int): The size of each chunk.
        
    Returns:
        list: A list of text chunks.
    """
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]



sentences = text.split('\n')
return sentences
processed_text8 = load_text8_dataset()

## tokenize text8


with open("text8.txt") as f:
    raw_text = f.read()

token_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]

chunks = [token_ids[i:i+512] for i in range(0, len(token_ids), 512)]

chunks = []
tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

for i in range(0, len(tokens), 512):
    chunk = tokens[i:i+512]
    chunks.append(chunk)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokens = tokenizer.tokenize(text)  # → list of subword pieces
ids = tokenizer.convert_tokens_to_ids(tokens)  # → list of integers

# Or in one go:
ids = tokenizer(text, add_special_tokens=False)["input_ids"]








text8_tokens = set()
for sentence in processed_text8[:10000]:  # optional slice while testing
    text = " ".join(sentence)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    text8_tokens.update(ids)

print("Text8 token count:", len(text8_tokens))
vocab_dict = tokenizer.get_vocab()
print(len(vocab_dict)) 


tokens = tokenizer.convert_ids_to_tokens(list(token_ids))


# Tokenize MS MARCO passages
msmarco_tokens = set()
for passage_list in ms_marco_passages:
    for passage in passage_list:
        text = passage['text']  # assuming structure like [{'text': "...", ...}]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        msmarco_tokens.update(ids)

# Combine & get size
combined_vocab = text8_tokens.union(msmarco_tokens)
print("Combined vocab size:", len(combined_vocab))