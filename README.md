Perfect! Here's a **Quick Start README section** for your teammate to get up and running immediately:

---

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

### 3️⃣ Make sure data and FAISS index exist

* Preprocessed CSV: `data/processed/processed_20news_train.csv`
* FAISS index: path in `utils/config.py` (`FAISS_INDEX_PATH`)
* Document IDs pickle: path in `utils/config.py` (`DOC_IDS_PATH`)

### 4️⃣ Run baseline evaluation

```bash

python -m utils.data_utils
python -m models.centralized_model
python -m retrieval.build_index
python -m experiments.exp_baseline
```


