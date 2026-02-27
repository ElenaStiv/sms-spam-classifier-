!pip install transformers torch scikit-learn

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

x = df["message"].tolist()
y = df["label"].tolist()

print("Dataset size:", len(df))
print(df.head())


# ============================================================
# 2. Tokenization
# ============================================================

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

encodings = tokenizer(
    x,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
)

labels = torch.tensor(y)


# ============================================================
# 3. PyTorch Dataset
# ============================================================

class SpamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ============================================================
# 4. Train/Test Split
# ============================================================

train_idx, test_idx = train_test_split(
    range(len(y)), test_size=0.2, shuffle=True, stratify=y
)

train_dataset = SpamDataset(
    {k: v[train_idx] for k, v in encodings.items()},
    labels[train_idx]
)

test_dataset = SpamDataset(
    {k: v[test_idx] for k, v in encodings.items()},
    labels[test_idx]
)


# ============================================================
# 5. DataLoaders
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# ============================================================
# 6. Load Model
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)


# ============================================================
# 7. Training Loop
# ============================================================

EPOCHS = 3

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")


# ============================================================
# 8. Evaluation
# ============================================================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)

accuracy = correct / total
print(f"\nTest Accuracy: {accuracy:.4f}")


# ============================================================
# 9. Prediction Function
# ============================================================

def predict(text):
    model.eval()
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return "spam" if pred == 1 else "ham"


print("\nPrediction example:")
print("Message: 'New Job opportunity for you in Dubai'")
print("Prediction:", predict("New Job opportunity for you in Dubai"))


# ============================================================
# 10. Save Model + Metadata
# ============================================================

os.makedirs("model", exist_ok=True)

model.save_pretrained("./model/clf")
tokenizer.save_pretrained("./model/clf")

with open("./model/info.pkl", "wb") as f:
    pickle.dump((MODEL_NAME, MAX_LEN), f)


# ============================================================
# 11. Load Model + Predict Again
# ============================================================

loaded_model = DistilBertForSequenceClassification.from_pretrained("./model/clf")
loaded_model.to(device)

loaded_tokenizer = DistilBertTokenizer.from_pretrained("./model/clf")

model_name, max_len = pickle.load(open("./model/info.pkl", "rb"))

print("\nReloaded model prediction:")
print(predict("NEw Job opportunity for you in Dubai"))