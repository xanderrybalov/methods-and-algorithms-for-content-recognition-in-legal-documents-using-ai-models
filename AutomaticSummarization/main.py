import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Клас для датасету
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

# Ініціалізація токенізатора та моделі T5 для узагальнення
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Завантаження та підготовка даних
file_path_legal = 'dumps/LegalDump.csv'
file_path_unlaw = 'dumps/DumpUnlaw.csv'
legal_df = pd.read_csv(file_path_legal)
unlaw_df = pd.read_csv(file_path_unlaw)

def generate_summary(text):
    print(f"Узагальнення тексту...")
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Узагальнення завершено.")
    return summary

# Узагальнення для legal_df
for idx in range(len(legal_df)):
    print(f"Узагальнення тексту {idx + 1} з {len(legal_df)} (Юридичний)...")
    legal_df.at[idx, 'summary'] = generate_summary(legal_df.at[idx, 'QouteText'])

# Узагальнення для unlaw_df
for idx in range(len(unlaw_df)):
    print(f"Узагальнення тексту {idx + 1} з {len(unlaw_df)} (Неюридичний)...")
    unlaw_df.at[idx, 'summary'] = generate_summary(unlaw_df.at[idx, 'QouteText'])

# Об'єднання даних
df = pd.concat([legal_df, unlaw_df])
df['label'] = df['is_legal']

# Розділення даних на навчальні та тестові
train_texts, test_texts, train_labels, test_labels = train_test_split(df['summary'], df['label'], test_size=0.2)

# Ініціалізація токенізатора та моделі DistilBert
dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Створення наборів даних для тренування та тестування
train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), dbert_tokenizer)
test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), dbert_tokenizer)

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
    model=dbert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

