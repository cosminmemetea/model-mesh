from transformers import AdamW, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from datetime import datetime

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def plot_metrics(metrics, output_dir):
    # Extract epochs and metrics
    epochs = metrics["epochs"]
    accuracy = metrics["accuracy"]
    f1_score = metrics["f1"]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy, label='Accuracy', marker='o')
    plt.plot(epochs, f1_score, label='F1 Score', marker='s')

    # Add labels, title, and legend
    plt.xlabel('Epochs')
    plt.xticks(epochs)  # Ensure only integer ticks for epochs
    plt.ylabel('Metrics')
    plt.title('Model Performance Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot with a unique name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = os.path.join(output_dir, f"model_performance_{timestamp}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"Performance plot saved to {image_path}")

def plot_confusion_matrix(predictions, labels, output_dir, class_names=None):
    # Generate confusion matrix
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')

    # Save the plot with a unique name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"Confusion matrix plot saved to {image_path}")

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
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    metrics = {"epochs": [], "accuracy": [], "f1": []}

    optimizer_grouped_parameters = [
        {"params": model.roberta.embeddings.parameters(), "lr": 1e-5},
        {"params": model.roberta.encoder.layer[:6].parameters(), "lr": 2e-5},
        {"params": model.roberta.encoder.layer[6:].parameters(), "lr": 3e-5},
        {"params": model.classifier.parameters(), "lr": 3e-5},
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # Use RAdam
    )

    # Train the model and collect metrics dynamically
    for epoch in range(training_args.num_train_epochs):
        print(f"Training epoch {epoch + 1}")
        trainer.train()  # Train for one epoch
        eval_results = trainer.evaluate()  # Evaluate after epoch
        metrics["epochs"].append(epoch + 1)  # Append epoch number
        metrics["accuracy"].append(eval_results["eval_accuracy"])  # Append accuracy
        metrics["f1"].append(eval_results["eval_f1"])  # Append F1 score

    # Final evaluation to collect predictions and labels
    predictions = trainer.predict(tokenized_datasets["validation"])
    pred_labels = predictions.predictions.argmax(axis=-1)
    true_labels = predictions.label_ids

    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

    # Plot metrics and confusion matrix
    plot_metrics(metrics, output_dir)
    plot_confusion_matrix(pred_labels, true_labels, output_dir, class_names=["Negative", "Neutral", "Positive"])

if __name__ == "__main__":
    fine_tune_cardiff_roberta()
