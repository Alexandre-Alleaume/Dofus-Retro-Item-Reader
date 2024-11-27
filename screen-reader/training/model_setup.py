# model_setup.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed").save_pretrained("/kaggle/working/trocr-processor")
VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").save_pretrained("/kaggle/working/trocr-model")

def load_model():
   

    # Load the fine-tuned model and processor from the previous training output directory
    model_dir = "/kaggle/working/output"  # Update this if the path is different
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    processor = TrOCRProcessor.from_pretrained(model_dir)

    # Configure model for text generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set generation parameters (customize if needed)
    model.config.max_length = 64
    model.config.no_repeat_ngram_size = 2
    model.config.early_stopping = True
    model.config.num_beams = 5

    return processor, model