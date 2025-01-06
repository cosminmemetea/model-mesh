from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def fine_tune_cardiff_roberta():
    # Define paths
    data_path = "/Users/mme/Workspace/mm/model-mesh/data/train/train.csv"
    validation_path = "/Users/mme/Workspace/mm/model-mesh/data/validated/validation.csv"
    output_dir = "/Users/mme/Workspace/mm/model-mesh/models/cardiff_roberta_finetuned"
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", num_labels=3)

    # Load dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": data_path, "validation": validation_path}
    )

    # Preprocess dataset
    def preprocess(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(preprocess, batched=True)

    # Define data collator (handles padding dynamically)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,  # Lower learning rate for fine-tuning
        weight_decay=0.01,  # Add regularization to prevent overfitting
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    fine_tune_cardiff_roberta()
