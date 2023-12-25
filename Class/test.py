from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification

from main import test_dataset, compute_metrics

# Завантаження збереженої моделі
model_path = "results/checkpoint-500" # Замініть шлях до вашої папки
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Оцінка моделі
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="/tmp/test_trainer",
        per_device_eval_batch_size=64,
    ),
    compute_metrics=compute_metrics
)

# Використання уже створеного test_dataset
test_results = trainer.evaluate(test_dataset)
print(test_results)
