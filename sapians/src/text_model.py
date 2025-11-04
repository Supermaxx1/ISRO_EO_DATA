from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def get_text_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.base_model(**inputs, output_hidden_states=True)
    # Use the last hidden state mean as embedding
    embeddings = outputs.hidden_states[-1].mean(dim=1)
    return embeddings
