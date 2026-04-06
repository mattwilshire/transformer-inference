import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Model:
    word_embeddings = np.fromfile("gpt2_weights/transformer_wte_weight.bin", dtype=np.float32).reshape(50257, 768)
    position_embeddings = np.fromfile("gpt2_weights/transformer_wpe_weight.bin", dtype=np.float32).reshape(1024, 768)

def embedding_layer(tokens):
    """Simply convert the tokens to embeddings, just a simple lookup."""
    return Model.word_embeddings[tokens]

def positional_embedding_layer(embeddings):
    """Adds the corresponding positional embeddings to the word embeddings"""
    return embeddings + Model.position_embeddings[:len(embeddings)]


def network(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer(text)['input_ids']
    embeddings = embedding_layer(tokens)
    position_embeddings = positional_embedding_layer(embeddings)
    print(position_embeddings)


def main():
    text = "What the"
    network(text)
    print()
    gpt2(text)


def gpt2(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    tokens = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    embedding_output = outputs.hidden_states[0].numpy()
    print(embedding_output)

if __name__ == "__main__":
    main()
