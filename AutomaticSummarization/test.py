from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification
from main import test_dataset, compute_metrics  # Імпортуйте compute_metrics з main.py

# Завантаження збереженої моделі
model_path = "results/checkpoint-500"
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Ініціалізація Trainer з необхідними налаштуваннями для оцінки
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./test_train_result",
        per_device_eval_batch_size=64,
    ),
    compute_metrics=compute_metrics  # Додайте функцію compute_metrics тут
)

# Оцінка моделі на тестовому наборі даних
test_results = trainer.evaluate(test_dataset)
print(test_results)
