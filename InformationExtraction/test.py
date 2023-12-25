from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification
from main import test_dataset, compute_metrics

# Шлях до збереженої моделі
model_path = "results/checkpoint-500"  # шлях до збереженої моделі

# Завантаження збереженої моделі
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Ініціалізація Trainer з необхідними налаштуваннями для оцінки
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./test_train_result",  # Папка для зберігання результатів оцінки
        per_device_eval_batch_size=64,     # Розмір пакета для оцінки
    ),
    compute_metrics=compute_metrics  # Ви можете вказати функцію для обчислення метрик, якщо потрібно
)

# Оцінка моделі на тестовому наборі даних
test_results = trainer.evaluate(test_dataset)
print(test_results)
