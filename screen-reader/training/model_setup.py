# model_setup.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model():
    # Load the pre-trained processor and model for printed text
    processor = TrOCRProcessor.from_pretrained("training/output")
    model = VisionEncoderDecoderModel.from_pretrained("training/output")

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