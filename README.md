
````markdown
# Quick Start: Run Centralized IR Baseline

Follow these steps after cloning the repo:

### 1️⃣ Create a virtual environment and activate
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```
if some files still miss then , install it additionally
```bash
pip install nltk
pip install beir
pip install rank_bm25
pip install matplotlib
pip install protobuf
```

### 4️⃣ Run baseline evaluation

```bash

python -m scripts.prepare_data_and_indices
python -m scripts.evaluate
```

