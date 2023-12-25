#DistilBERT model
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Завантаження даних
file_path_legal = 'dumps/LegalDump.csv'  # Замініть на шлях до файлу з юридичними текстами
file_path_unlaw = 'dumps/DumpUnlaw.csv' # Замініть на шлях до файлу з неюридичними текстами

legal_df = pd.read_csv(file_path_legal)
unlaw_df = pd.read_csv(file_path_unlaw)

# Об'єднання даних та міткування
legal_df['label'] = 1
unlaw_df['label'] = 0
df = pd.concat([legal_df, unlaw_df])

# Токенізація та форматування даних
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Розділення даних на навчальні та тестові
train_texts, test_texts, train_labels, test_labels = train_test_split(df['QouteText'], df['is_legal'], test_size=0.2)

# Створення наборів даних для тренування та тестування
train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

# Ініціалізація моделі
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Налаштування параметрів тренування
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Тренування моделі
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
