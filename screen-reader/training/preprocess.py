# preprocess.py

import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor
import os
from dataset import ItemDataset  # Import the custom ItemDataset class

# Load your dataset JSON file
def load_dataset(json_path):
    with open(json_path, 'r') as file:
        dataset = json.load(file)
    print(f"Loaded dataset: {type(dataset)}")
    return dataset

# Function to create a DataLoader for the dataset
def create_dataloader(dataset, processor, image_dir, batch_size=8, shuffle=True):
    item_dataset = ItemDataset(dataset, image_dir, processor) 
    return DataLoader(item_dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage
if __name__ == "__main__":
    # Load processor and dataset
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    dataset_path = "dataset/dataset.json"  # Path to your JSON file
    image_dir = "dataset/images"  # Path to the directory with images

    # Load the dataset
    dataset = load_dataset(dataset_path)

    # Create DataLoader
    dataloader = create_dataloader(dataset, processor, image_dir)

    # Test loading a batch
    for batch in dataloader:
        images, labels = batch['image'], batch['labels']
        print("Batch of images shape:", images.shape)
        print("Batch of labels shape:", labels.shape)
        break