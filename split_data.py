import random

random.seed(42)

with open("./data/train.jsonl", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

n = len(lines)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

splits = {
    "train.jsonl": lines[:train_end],
    "valid.jsonl": lines[train_end:val_end],
    "test.jsonl": lines[val_end:],
}

for name, data in splits.items():
    with open(f"./data/{name}", "w", encoding="utf-8") as f:
        f.writelines(data)
    print(f"{name}: {len(data)} samples")
