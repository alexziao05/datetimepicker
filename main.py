"""
AutoClass API: 
Automatically infers the appropriate architecture for each task and machine learning framework 
based on the name or path to the pretrained weights and configuration file.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

# Use from_pretrained() to load the weights and configuration file from the Hub into the model and preprocessor class.

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# GPT-2 doesn't have a padding token, so we set it to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# device_map="auto" automatically allocates the model weights to fastest device first 
# dtype="auto" directly initializes the model wieghts in the data type they're stored in 

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype="auto", device_map="auto")

"""
### Inference Model ###
Tokenize the text and return PyTorch tensors with the tokenizer
Tokenizer - converts string into token IDs that the model can understand
return_tensors="pt" - returns PyTorch tensors instead of lists 
.to(model.device) - moves the tensors to the same device as the model (GPU or CPU)
"""

# model_inputs = tokenizer(["Who is the first president of the United States?"], return_tensors="pt").to(model.device)

# For inference, pass tokenized inputs to generate() to generate text
# Decode the token ids back into text with batch_decode() 

# generated_ids = model.generate(**model_inputs, max_new_tokens=50)
# print(tokenizer.batch_decode(generated_ids)[0])


"""
### Trainer ###
Trainer is a complete training and evaluation loop for PyTorch models. 
It abstracts away a lot of the boilerplate usually involved in manually writing a training loop, so you can start training faster and focus on training design choices. 
You only need a model, dataset, a preprocessor, and a data collator to build batches of data from the dataset.
"""

time_expressions_dataset = load_dataset("namesarnav/time_expressions_dataset", split="train")
time_expressions_dataset_split = time_expressions_dataset.train_test_split(test_size=0.2)
validation_dataset = time_expressions_dataset_split["test"]

def tokenize_dataset_to_tensors(dataset): 
    # Tokenize input_text and target_output, and set labels for causal LM
    input_encodings = tokenizer(dataset["input_text"], truncation=True, padding="max_length", max_length=128)
    label_encodings = tokenizer(dataset["target_output"], truncation=True, padding="max_length", max_length=128)
    input_encodings["labels"] = label_encodings["input_ids"]
    return input_encodings

# Apply this function to the entire dataset using map()
time_expressions_dataset = time_expressions_dataset.map(tokenize_dataset_to_tensors, batched=True)
validation_dataset = validation_dataset.map(tokenize_dataset_to_tensors, batched=True)

# Load the data collator to build batches of data
"""
### What is a data collator? ###
When you train a model, you DO NOT want to pass one example at a time to the model.
Instead, you want to pass a batch of examples to the model at once.
A data collator is a function that takes a list of examples and combines them into a batch.
"""

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Defining training arguments 
training_args = TrainingArguments(
    # output directory to save model checkpoints and other outputs
    output_dir = "./results",         
    # An epoch represents one complete pass through the entire training dataset 
    eval_strategy = "epoch", 
    # The batch size determines how many steps are needed to complete one epoch
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    # Gradient accumulation to simulate larger batch size
    gradient_accumulation_steps=8,
    # How many times the model will see the entire training dataset during training
    num_train_epochs=5,
    # In training, models can "memorize" the training data too well, leading to poor performance on new data.
    # Weight decay is a regularization technique that helps prevent overfitting by adding a penalty
    # What is penalty? It discourages the model from assigning too much importance to any single feature in the training data.
    weight_decay=0.01,
)

# Define the trainer 
trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=time_expressions_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Evaluate the model
print(trainer.evaluate())

# Save the model and tokenizer
trainer.save_model("./fine_tuned_time_expressions_model")
tokenizer.save_pretrained("./fine_tuned_time_expressions_model")
