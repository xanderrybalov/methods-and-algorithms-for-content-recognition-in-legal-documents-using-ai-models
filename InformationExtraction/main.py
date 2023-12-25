import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Ініціалізація моделі spaCy для витягання інформації
nlp = spacy.load("en_core_web_sm")

def extract_information(text):
    doc = nlp(text)
    # Витягання іменованих сутностей, наприклад
    features = " ".join([ent.text for ent in doc.ents])
    return features

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

# Завантаження та підготовка даних
file_path_legal = 'dumps/LegalDump.csv'
file_path_unlaw = 'dumps/DumpUnlaw.csv'
legal_df = pd.read_csv(file_path_legal)
unlaw_df = pd.read_csv(file_path_unlaw)

# Об'єднання даних
df = pd.concat([legal_df, unlaw_df])

# Витягання інформації для обох наборів даних
df['extracted_info'] = df['QouteText'].apply(extract_information)

# Розділення даних на навчальні та тестові
train_texts, test_texts, train_labels, test_labels = train_test_split(df['extracted_info'], df['is_legal'], test_size=0.2)

# Ініціалізація токенізатора DistilBert
dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Створення наборів даних для тренування та тестування
train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), dbert_tokenizer)
test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), dbert_tokenizer)

# Ініціалізація моделі DistilBert для класифікації
dbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

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

