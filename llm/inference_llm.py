import os
import atexit
from pathlib import Path
import shutil
import transformers 
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run LLM chatbot")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Local path or Hub model name")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    str(Path(args.model_name_or_path).expanduser()),
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_name_or_path).expanduser()))

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)


# Step 2: Define a function to interact with the chatbot
def interact_with_chatbot(user_input, conversation_history):
    # Step 2.1: Add user input to the conversation history
    conversation_history.append(f"User: {user_input}")
    
    # Step 2.2: Prepare the input text for the model
    conversation_text = " ".join(conversation_history[-5:])  # Use only the last 5 exchanges to keep context manageable
    
    # Step 2.3: Generate a response using the chatbot pipeline

    messages = [
        {"role": "system", "content": "You are a chemist that tries to answer user questions as accurately as possible. Work through problems given to you step by step. If you don't know the answer, you can say I don't know."},
        {"role": "user", "content": conversation_text},
    ]

    outputs = pipeline(messages, max_new_tokens=1024)
    response_text = outputs[0]["generated_text"][-1]
    
    return response_text

# Step 3: Define a function to delete the model files from the cache directory
def delete_model_files():
    if args.model_name_or_path.startswith("microsoft/") or args.model_name_or_path.startswith("huggingface/"):
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{args.model_name_or_path.replace('/', '--')}")
    else:
        print("Skipping model file deletion for local models.")
        return
    
    if os.path.exists(cache_dir):
        user_input = input("Do you want to delete the model files from the cache directory? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(cache_dir)
            print(f"Deleted model files from cache directory: {cache_dir}")
        else:
            print("Model files not deleted from cache directory.")
    else:
        print(f"Model files not found in cache directory: {cache_dir}")

# Step 4: Register the delete_model_files function to be called on program exit
atexit.register(delete_model_files)

# Step 5: Start the conversation loop
print(f"Welcome to the MSDC Chatbot! (Model: {args.model_name_or_path})")
print("Type 'quit' to end the conversation.\n")

conversation_history = []

while True:
    # Step 5.1: Get user input
    user_input = input("User: ")
    
    # Step 5.2: Check if the user wants to quit
    if user_input.lower() == 'quit':
        print("Thank you for using the MSDC Chatbot. Goodbye!")
        break
    
    # Step 5.3: Generate and print the chatbot's response
    response = interact_with_chatbot(user_input, conversation_history)
    print(f"Chatbot: {response}") 