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

    # token and position embeddings
    word_embeddings = np.fromfile("gpt2_weights/transformer_wte_weight.bin", dtype=np.float32).reshape(VOCAB_SIZE, EMB_DIM)
    position_embeddings = np.fromfile("gpt2_weights/transformer_wpe_weight.bin", dtype=np.float32).reshape(N_SEQ_LENGTH, EMB_DIM)

    # final layer weights
    ln_f_weight = np.fromfile("gpt2_weights/transformer_ln_f_weight.bin", dtype=np.float32)
    ln_f_bias = np.fromfile("gpt2_weights/transformer_ln_f_bias.bin", dtype=np.float32)

    # per layer weights
    ln_1_weight = []
    ln_1_bias = []
    ln_2_weight = []
    ln_2_bias = []
    attn_c_attn_weight = []
    attn_c_attn_bias = []
    attn_c_proj_weight = []
    attn_c_proj_bias = []
    mlp_c_fc_weight = []
    mlp_c_fc_bias = []
    mlp_c_proj_weight = []
    mlp_c_proj_bias = []

    for i in range(N_LAYER):
        path = f"gpt2_weights/transformer_h_{i}_"

        # attention layer norms
        ln_1_weight.append(np.fromfile(f"{path}ln_1_weight.bin", dtype=np.float32))
        ln_1_bias.append(np.fromfile(f"{path}ln_1_bias.bin", dtype=np.float32))
        ln_2_weight.append(np.fromfile(f"{path}ln_2_weight.bin", dtype=np.float32))
        ln_2_bias.append(np.fromfile(f"{path}ln_2_bias.bin", dtype=np.float32))

        # attention Q/K/V combined projection and output projection
        attn_c_attn_weight.append(np.fromfile(f"{path}attn_c_attn_weight.bin", dtype=np.float32).reshape(EMB_DIM, 3 * EMB_DIM))
        attn_c_attn_bias.append(np.fromfile(f"{path}attn_c_attn_bias.bin", dtype=np.float32))
        attn_c_proj_weight.append(np.fromfile(f"{path}attn_c_proj_weight.bin", dtype=np.float32).reshape(EMB_DIM, EMB_DIM))
        attn_c_proj_bias.append(np.fromfile(f"{path}attn_c_proj_bias.bin", dtype=np.float32))

        # MLP, expands to 3072 and then project back to 768
        mlp_c_fc_weight.append(np.fromfile(f"{path}mlp_c_fc_weight.bin", dtype=np.float32).reshape(EMB_DIM, N_MLP_HIDDEN))
        mlp_c_fc_bias.append(np.fromfile(f"{path}mlp_c_fc_bias.bin", dtype=np.float32))
        mlp_c_proj_weight.append(np.fromfile(f"{path}mlp_c_proj_weight.bin", dtype=np.float32).reshape(N_MLP_HIDDEN, EMB_DIM))
        mlp_c_proj_bias.append(np.fromfile(f"{path}mlp_c_proj_bias.bin", dtype=np.float32))



def embedding_layer(tokens):
    """Simply convert the tokens to embeddings, just a simple lookup."""
    return Model.word_embeddings[tokens]

def positional_embedding_layer(embeddings):
    """Adds the corresponding positional embeddings to the word embeddings"""
    return embeddings + Model.position_embeddings[:len(embeddings)]

def layer_norm(embeddings, weight, bias, eps=1e-5):
    mean = np.mean(embeddings, axis=-1, keepdims=True)
    var = np.var(embeddings, axis=-1, keepdims=True)
    normalized = (embeddings - mean) / np.sqrt(var + eps)
    return weight * normalized + bias

def transformer_block(x, layer_idx):
    normed = layer_norm(x, Model.ln_1_weight[layer_idx], Model.ln_1_bias[layer_idx])
    print(normed)
    pass


def network(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer(text)['input_ids']
    embeddings = embedding_layer(tokens)
    position_embeddings = positional_embedding_layer(embeddings)

    for layer_idx in range(Model.N_LAYER):
        transformer_block(position_embeddings, layer_idx)
        break


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
        # Grab the embedding output
        embeds = model.transformer.wte(tokens['input_ids']) + model.transformer.wpe(torch.arange(len(tokens['input_ids'][0])))

        # Run it through the first block's layer norm
        ln_1_output = model.transformer.h[0].ln_1(embeds)
        print("After ln_1:")
        print(ln_1_output.numpy())

if __name__ == "__main__":
    main()
