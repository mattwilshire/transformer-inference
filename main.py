import numpy as np
from transformers import GPT2Tokenizer

class Model:
    word_embeddings = np.fromfile("gpt2_weights/transformer_wte_weight.bin", dtype=np.float32).reshape(50257, 768)

def embedding_layer(tokens):
    """Simply convert the tokens to embeddings, just a simple lookup."""
    return Model.word_embeddings[tokens]

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer("What the")['input_ids']

    embeddings = embedding_layer(tokens)
    print(embeddings.shape, embeddings)


if __name__ == "__main__":
    main()
