# MLXTwoTowers
Week 2 coursework - Search and Retrieval


## Tyrone's implemenation

```bash
python preprocess_data.py 
python train_models.py
# need to have a Redis server running
python store_documents.py
uvicorn app.main:app --reload
```