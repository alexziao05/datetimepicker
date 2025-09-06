"""
AutoClass API: 
Automatically infers the appropriate architecture for each task and machine learning framework 
based on the name or path to the pretrained weights and configuration file.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Use from_pretrained() to load the weights and configuration file from the Hub into the model and preprocessor class.

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# device_map="auto" automatically allocates the model weights to fastest device first 
# dtype="auto" directly initializes the model wieghts in the data type they're stored in 

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="auto", device_map="auto")

"""
Tokenize the text and return PyTorch tensors with the tokenizer
Tokenizer - converts string into token IDs that the model can understand
return_tensors="pt" - returns PyTorch tensors instead of lists 
.to(model.device) - moves the tensors to the same device as the model (GPU or CPU)
"""

model_inputs = tokenizer(["Who is the first president of the United States?"], return_tensors="pt").to(model.device)

# For inference, pass tokenized inputs to generate() to generate text
# Decode the token ids back into text with batch_decode() 

generated_ids = model.generate(**model_inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids)[0])
