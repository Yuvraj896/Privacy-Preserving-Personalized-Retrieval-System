#!/usr/bin/env python3
"""
Download 20 Newsgroups and make NUM_CLIENTS non-IID client partitions.
Each client_i folder will contain pairs.csv -> query, doc, label
Run (from project root):
    python scripts/data_prep_newsgroups.py --num_clients 20 --out_dir data
"""

import sys
from pathlib import Path

# make project root importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv
import random
from sklearn.datasets import fetch_20newsgroups
from config import NUM_CLIENTS, DOCS_PER_CLIENT, PAIRS_PER_CLIENT

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def make_pairs_for_client(client_docs, target_names, pairs_per_client=1000):
    pairs = []
    docs_by_label = {}
    for text, label in client_docs:
        docs_by_label.setdefault(label, []).append(text)

    all_labels = list(range(len(target_names)))
    for _ in range(pairs_per_client):
        fav_labels = list(docs_by_label.keys())
        if random.random() < 0.8 and fav_labels:
            q_label = random.choice(fav_labels)
        else:
            q_label = random.choice(all_labels)

        query = target_names[q_label]

        # positive
        if docs_by_label.get(q_label):
            pos_doc = random.choice(docs_by_label[q_label])
            pairs.append((query, pos_doc, 1))

        # negative
        neg_label = random.choice([l for l in all_labels if l != q_label])
        neg_doc_candidates = docs_by_label.get(neg_label, [])
        if neg_doc_candidates:
            neg_doc = random.choice(neg_doc_candidates)
        else:
            other_docs = [t for t, l in client_docs if l != q_label]
            neg_doc = random.choice(other_docs)
        pairs.append((query, neg_doc, 0))

    return pairs

def main(num_clients: int, out_dir: Path):
    print("⬇️ Downloading 20 Newsgroups dataset …")
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    docs, labels, target_names = data.data, data.target, data.target_names

    grouped = {}
    for txt, lbl in zip(docs, labels):
        grouped.setdefault(lbl, []).append(txt)

    n_labels = len(target_names)
    random.seed(42)

    for cid in range(1, num_clients + 1):
        fav1 = cid % n_labels
        fav2 = (cid * 3) % n_labels
        favs = [fav1] if fav1 == fav2 else [fav1, fav2]

        client_docs = []
        num_from_favs = int(DOCS_PER_CLIENT * 0.75)
        per_fav = max(1, num_from_favs // len(favs))
        for f in favs:
            src = grouped.get(f, [])
            sample = random.sample(src, min(len(src), per_fav))
            client_docs.extend([(s, f) for s in sample])

        while len(client_docs) < DOCS_PER_CLIENT:
            lbl = random.randrange(n_labels)
            src = grouped.get(lbl, [])
            if src:
                client_docs.append((random.choice(src), lbl))

        pairs = make_pairs_for_client(client_docs, target_names, PAIRS_PER_CLIENT)

        client_dir = out_dir / f"client_{cid}"
        ensure_dir(client_dir)
        csv_path = client_dir / "pairs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query", "doc", "label"])
            for q, d, l in pairs:
                w.writerow([q, d.replace("\n", " "), l])
        print(f"✅ Saved {csv_path}")

    # global validation
    val_dir = out_dir / "validation"
    ensure_dir(val_dir)
    val_csv = val_dir / "val.csv"
    val_pairs = []
    for lbl, texts in grouped.items():
        for t in random.sample(texts, min(10, len(texts))):
            val_pairs.append((target_names[lbl], t, 1))
            neg_lbl = (lbl + 1) % n_labels
            val_pairs.append((target_names[lbl], random.choice(grouped[neg_lbl]), 0))

    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "doc", "label"])
        for q, d, l in val_pairs:
            w.writerow([q, d.replace("\n", " "), l])
    print(f"✅ Saved validation set → {val_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--out_dir", type=str, default="../data")
    args = parser.parse_args()
    main(args.num_clients, Path(args.out_dir))
