import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Model:
    # ran gpt2_export to get these
    VOCAB_SIZE = 50257
    N_SEQ_LENGTH = 1024
    EMB_DIM = 768
    HEAD_DIM = 64
    N_LAYER = 12
    N_HEAD = 12
    N_MLP_HIDDEN = 3072


    word_embeddings = np.fromfile("gpt2_weights/transformer_wte_weight.bin", dtype=np.float32).reshape(VOCAB_SIZE, EMB_DIM)
    position_embeddings = np.fromfile("gpt2_weights/transformer_wpe_weight.bin", dtype=np.float32).reshape(N_SEQ_LENGTH, EMB_DIM)

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

    for i in range(Model.N_LAYER):
        print(i)

    print(position_embeddings)


def main():
    text = "What the"
    network(text)
    print()
    # gpt2(text)


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
