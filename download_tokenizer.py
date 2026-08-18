from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-en-indic-1B",
    trust_remote_code=True,
)

tokenizer.save_pretrained("models/eng_to_prakrit_2-final")