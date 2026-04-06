import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
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
    """Normalize embeddings to counter vanishing gradients"""
    mean = np.mean(embeddings, axis=-1, keepdims=True)
    var = np.var(embeddings, axis=-1, keepdims=True)
    normalized = (embeddings - mean) / np.sqrt(var + eps)
    return weight * normalized + bias

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(x, layer_idx):
    seq_len = x.shape[0]

    # one matmul to get each tokens QKV, will be (2, 2304)
    # QKV are concatendated together so Q = 768 + K = 768 + V = 768 == 2304
    qkv = x @ Model.attn_c_attn_weight[layer_idx] + Model.attn_c_attn_bias[layer_idx]

    # print(qkv)
    # print(qkv.shape)

    # split into Q, K, V each
    q, k, v = np.split(qkv, 3, axis=-1)

    # print(q.shape)
    # print(q)

    # casual mask is used to prevent attending to future tokens by setting them to negative infinity so softmax sets to them 0 which makes them useless in weighted sum
    # triu makes a triangle e.g
    #     [[0, 1, 1, 1],
    #     [0, 0, 1, 1],
    #     [0, 0, 0, 1],
    #     [0, 0, 0, 0]]
    # The ones become -1e10
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e10

    # process each head independently, should be done in parallel but this is for learning purposes
    head_outputs = []
    for h in range(Model.N_HEAD):
        # pull out q k v for the current head, will each be (seq_len, 64)
        q_h = q[:, h *  Model.HEAD_DIM : (h + 1) * Model.HEAD_DIM]
        k_h = k[:, h * Model.HEAD_DIM : (h + 1) * Model.HEAD_DIM]
        v_h = v[:, h * Model.HEAD_DIM : (h + 1) * Model.HEAD_DIM]

        # attention scores: (seq_len, 64) @ (64, seq_len) = (seq_len, seq_len)
        # Q @ K for every token
        scores = q_h @ k_h.T / np.sqrt(Model.HEAD_DIM)
        # add scores to the mask, scores we aren't allowed to see will be large negative numbers as described previously
        scores = scores + mask

        # get probabiliets, used for weighted sum, e.g 0.9 will mean a token attends heavily to the current token allowing it to consume more of it's V
        score_probs = softmax(scores)

        # weighted sum of values, tokens that attended more will have higher scores offering more of their V
        head_output = score_probs @ v_h # (seq_len, 64)
        # print(head_output.shape)
        head_outputs.append(head_output)
    
    # concatenate all heads, 12 x (seq_len, 64) -> (seq_len, 768)
    attn_out = np.concatenate(head_outputs, axis=-1)
    # print(attn_out.shape)

    # output projection
    return attn_out @ Model.attn_c_proj_weight[layer_idx] + Model.attn_c_proj_bias[layer_idx]


def gelu(x):
    # smoother ReLU - instead of a hard cutoff, it gradually tapers negative values toward zero
    # https://miro.medium.com/v2/resize:fit:720/format:webp/0*jetafLazYuwIXGuH.png
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def feed_forward(x, layer_idx):
    # expands (seq_len, 768) @ (768, 3072) = (seq_len, 3072)
    hidden = x @ Model.mlp_c_fc_weight[layer_idx] + Model.mlp_c_fc_bias[layer_idx]
    hidden = gelu(hidden)
    # project back (seq_len, 3072) @ (3072, 768) = (seq_len, 768)
    return hidden @ Model.mlp_c_proj_weight[layer_idx] + Model.mlp_c_proj_bias[layer_idx]
    

def transformer_block(x, layer_idx):
    # normalize the input so values are stable before attention, weights and biases are used for this normalization
    normed = layer_norm(x, Model.ln_1_weight[layer_idx], Model.ln_1_bias[layer_idx])

    # each token looks at other tokens to figure out what's relevant to it building embeddings with rich context
    attn_out = self_attention(normed, layer_idx)
    
    # Add the original input back - this "residual connection" lets gradients
    # flow straight through during training and means the attention only needs
    # to learn what to ADD to the representation, not rebuild it from scratch
    x = x + attn_out

    # Normalize again before the feed-forward step
    normed = layer_norm(x, Model.ln_2_weight[layer_idx], Model.ln_2_bias[layer_idx])

    # Each token independently processes its own representation through a
    # wider hidden layer (768 -> 3072 -> 768) - this is where the network
    # does most of its "thinking" / "understanding" about individual token meaning
    ff_out = feed_forward(normed, layer_idx)
    # residual
    x = x + ff_out
    return x


def network(text) -> int:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer(text)['input_ids']

    # -- network part ---
    embeddings = embedding_layer(tokens)
    position_embeddings = positional_embedding_layer(embeddings)
    x = position_embeddings

    for layer_idx in range(Model.N_LAYER):
        x = transformer_block(x, layer_idx)
    
    # Final layer norm
    x = layer_norm(x, Model.ln_f_weight, Model.ln_f_bias)

    # print(x.shape)

    # project to vocabulary: (seq_len, 768) @ (768, 50257) = (seq_len, 50257)
    logits = x @ Model.word_embeddings.T

    # get highest scoring token
    next_token = np.argmax(logits[-1])
    return next_token


def main(text, max_tokens):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"Prompt: {text}", end='')
    for _ in range(max_tokens):
        token = network(text)
        token_string = tokenizer.decode(token)
        print(token_string, end='', flush=True)
        text += tokenizer.decode(token)
    print()


if __name__ == "__main__":
    prompt = "What is a solar eclipse?"
    max_tokens = 19
    main(prompt, max_tokens)

# def gpt2(text):
#     # not used anymore, was used to understand the outputs at each layer and ensure manual network was correct
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#     model.eval()
#     tokens = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         embeds = model.transformer.wte(tokens['input_ids']) + model.transformer.wpe(torch.arange(len(tokens['input_ids'][0])))
#         ln_1_output = model.transformer.h[0].ln_1(embeds)
#         print("After ln_1:")
#         print(ln_1_output.numpy())
