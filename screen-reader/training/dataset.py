import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


class ItemDataset(Dataset):
    def __init__(self, data, image_dir, processor):
        self.data = data  # Now, data can be directly passed as a list.
        self.image_dir = image_dir
        self.processor = processor


    def __len__(self):
        # Number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get data entry at the specified index
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["imageUrl"])
        
        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Process image using the TrOCR processor
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()

        # Convert item data (name, attributes) into a target string
        target_text = self.format_target_text(item["itemData"])

        # Encode the target text
        labels = self.processor.tokenizer(target_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids.squeeze()

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    def format_target_text(self, item_data):
        """
        Formats item_data dictionary into a target string for TrOCR.
        This target string should contain all item attributes and values.
        Example output: "Talisman d'Elya Wood +150 Vitalité +32 Intelligence +40 Sagesse ..."
        """
        # Extract the name
        name = item_data.get("name", "")
        
        # Extract and format characteristics as a single string
        attributes = [f"{value} {key.replace('_', ' ')}" for key, value in item_data.items() if key != "name"]
        attributes_text = " ".join(attributes)
        
        # Combine name and attributes
        target_text = f"{name} {attributes_text}"
        
        return target_text