# train.py

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from preprocess import load_dataset, create_dataloader
from model_setup import load_model
from dataset import ItemDataset
from torch.utils.data import random_split
import os

def main():
    # Set paths and hyperparameters
    dataset_path = "dataset/dataset.json"  # JSON data file
    image_dir = "dataset/images"  # Image folder
    output_dir = "training/output"  # Model output directory
    batch_size = 8
    num_epochs = 10
    learning_rate = 3e-5

    # Load dataset and model
    processor, model = load_model()
    dataset = load_dataset(dataset_path)

    # Get the list of data entries (not the whole dataset dictionary)
    data = dataset["data"]

    # Split dataset into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"Train dataset length: {len(train_data)}")
    print(f"Validation dataset length: {len(val_data)}")


  
    train_dataset = ItemDataset(train_data, image_dir, processor)
    eval_dataset = ItemDataset(val_data, image_dir, processor)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        learning_rate=learning_rate,
        predict_with_generate=True
    )

    # Define trainer with validation dataset
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add validation dataset here
        tokenizer=processor.feature_extractor,
    )

    # Start training
    trainer.train()

    # Save final model and processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()