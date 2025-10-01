import numpy as np

# 1. Utilities
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def gelu(x):
    # approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def glorot_uniform(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


# 2. Token Embedding
class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # kecilkan inisialisasi
        self.weight = (np.random.randn(vocab_size, d_model) * 0.02).astype(np.float32)
    def __call__(self, token_ids):
        # token_ids: (B, T) integers
        return self.weight[token_ids]   # returns (B, T, D)


# 3. Positional Encoding
def sinusoidal_pos_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len)[:, None]    # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe  # (max_len, d_model)


# 4. Causal Mask
def causal_mask(T):
    # True where we should mask (j > i)
    return np.triu(np.ones((T, T), dtype=bool), k=1)

# 5. Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q,K,V: (B, H, T, d_k)
    dk = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(dk)   # (B,H,T,T)
    if mask is not None:
        scores = np.where(mask[np.newaxis, np.newaxis, :, :], -1e9, scores)
    weights = softmax(scores, axis=-1)   # (B,H,T,T)
    out = np.matmul(weights, V)          # (B,H,T,d_k)
    return out, weights

# 6. Multi-Head Attention
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # weight matrices
        self.Wq = glorot_uniform(d_model, d_model)
        self.Wk = glorot_uniform(d_model, d_model)
        self.Wv = glorot_uniform(d_model, d_model)
        self.Wo = glorot_uniform(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        B, T, D = x.shape
        Q = x @ self.Wq       # (B,T,D)
        K = x @ self.Wk
        V = x @ self.Wv
        # reshape -> (B, H, T, d_k)
        Q = Q.reshape(B, T, self.n_heads, self.d_k).transpose(0,2,1,3)
        K = K.reshape(B, T, self.n_heads, self.d_k).transpose(0,2,1,3)
        V = V.reshape(B, T, self.n_heads, self.d_k).transpose(0,2,1,3)
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (B,H,T,d_k) -> concat heads
        out = attn_out.transpose(0,2,1,3).reshape(B, T, D)  # (B,T,D)
        out = out @ self.Wo
        return out, attn_weights  # out: (B,T,D); attn_weights: (B,H,T,T)

# 7. LayerNorm
class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta = np.zeros((d_model,), dtype=np.float32)
    def __call__(self, x):
        # x: (B,T,D)
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# 8. FeedForward
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = glorot_uniform(d_model, d_ff)
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = glorot_uniform(d_ff, d_model)
        self.b2 = np.zeros((d_model,), dtype=np.float32)
    def __call__(self, x):
        # x: (B,T,D)
        x1 = x @ self.W1 + self.b1     # (B,T,D_ff)
        x2 = gelu(x1)
        out = x2 @ self.W2 + self.b2   # (B,T,D)
        return out

# 9. TransformerBlock
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        # x: (B,T,D)
        y = self.ln1(x)
        mha_out, attn = self.mha.forward(y, mask)
        x = x + mha_out               # residual
        z = self.ln2(x)
        ffn_out = self.ffn(z)
        x = x + ffn_out               # residual
        return x, attn

# 10. TransformerDecoder
class TransformerDecoder:
    def __init__(self, vocab_size, max_len, n_layers, d_model, n_heads, d_ff):
        self.embed = TokenEmbedding(vocab_size, d_model)  # (V,D)
        self.pos_emb = sinusoidal_pos_encoding(max_len, d_model)  # (max_len,D)
        self.layers = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_final = LayerNorm(d_model)
        self.vocab_size = vocab_size
        # linear to vocab
        self.W_out = glorot_uniform(d_model, vocab_size)
        self.b_out = np.zeros((vocab_size,), dtype=np.float32)
        # optional: tie weights -> self.W_out = self.embed.weight.T

    def forward(self, token_ids):
        # token_ids: (B,T)
        B, T = token_ids.shape
        x = self.embed(token_ids)         # (B,T,D)
        x = x + self.pos_emb[:T]          # broadcast (T,D) -> (B,T,D)
        # create causal mask once
        mask = causal_mask(T)             # (T,T)
        attn_weights_list = []
        for layer in self.layers:
            x, attn = layer.forward(x, mask)
            attn_weights_list.append(attn)
        x = self.ln_final(x)              # (B,T,D)
        logits = x @ self.W_out + self.b_out  # (B,T,V)
        return logits, attn_weights_list

