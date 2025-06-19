# MLXTwoTowers
Week 2 coursework - Search and Retrieval


## Tyrone's implemenation

To get a complete server running end to end, you should have a Computa server.

### Setup the environment

```bash
# on your machine
pip install -r requirements.txt
mkdir data
ssh -i <key-file> -p <computa-port> root@<computa-server>
# on Computa server
apt update
apt install -y git build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
<edit your .bashrc file as it says to>
bash
pyenv install 3.13.3
git clone https://github.com/AdamBeedell/MLXTwoTowers.git
cd MLXTwoTowers
pip install -r requirements.txt
mkdir data
```

### Running the core workflow

```bash
# On Computa server
python word2vec.py
# On your machine
scp -i <key-file> -P <computa-port> root@216.249.100.66:MLXTwoTowers/data/word2vec_skipgram.pth data/
python preprocess_data.py 
scp -i <key-file> -P <computa-port> data/datasets.pt root@216.249.100.66:MLXTwoTowers/data/
# On Computa server
python train_models.py
# On your machine
scp -i <key-file> -P <computa-port> root@216.249.100.66:MLXTwoTowers/data/models.pth data/
docker compose up --build
python store_documents.py
```

You can now visit http://localhost:8000/search?query=url+encoded+query. Click pretty-print to see it as nice JSON.
