import pickle
from transformers import BertTokenizer


ms_marco = pickle.load(open("msmarco_triplets_untokenized.pkl", "rb"))

ms_marco = [row for row in ms_marco if row.get("pos") and row.get("neg")]   #### Not sure what these are doing in the dataset tbh

ms_marco_tokenized = []
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)  ##### load a tokenizer which doesnt care about length of text. Many do because they want to do embeddings after.

i=0
for data in ms_marco:
    line = {
        "query_input_ids": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data['query'])),
        "pos_input_ids": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data['pos'])),
        "neg_input_ids": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data['neg']))
    }
    ms_marco_tokenized.append(line)
    i+=1
    percent = round(i/502938*100)
    if i%1000: 
        print(f"1000 lines tokenized, {percent}%")

with open("msmarco_triplets_tokenized_to_idx.pkl", "wb") as f:
    pickle.dump(ms_marco_tokenized, f)

