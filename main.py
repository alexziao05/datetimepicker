# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="auto", device_map="auto")

