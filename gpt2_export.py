import os
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
 
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
print()
print("GPT2 SUMMARY")
print("=" * 60)
 
config = model.config
print(f"  Vocab size:         {config.vocab_size}")
print(f"  Context length:     {config.n_positions}")
print(f"  Embedding dim:      {config.n_embd}")
print(f"  Num layers:         {config.n_layer}")
print(f"  Num attention heads: {config.n_head}")
print(f"  Head dim:           {config.n_embd // config.n_head}")
print(f"  Activation:         {config.activation_function}")
print()

for name, tensor in model.state_dict().items():
    print(f"  {name:50s}  shape={str(list(tensor.shape)):20s}  dtype={tensor.dtype}")
 
print()

weights_path = "./gpt2_weights"

os.makedirs(weights_path, exist_ok=True)

for name, tensor in model.state_dict().items():
    # convert to f32 numpy array (contiguous)
    arr = tensor.cpu().numpy().astype(np.float32)
 
    # sanitize filename: replace dots with underscores
    filename = name.replace(".", "_") + ".bin"
    filepath = os.path.join(weights_path, filename)
    arr.tofile(filepath)
print(f"Exported weights to {weights_path}")