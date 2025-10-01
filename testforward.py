import numpy as np
from transformer import TransformerDecoder, softmax

# Hyperparams kecil untuk test
B, T, V = 2, 5, 1000
D, H, D_ff, N = 64, 8, 256, 2
max_len = 128

# Init model
model = TransformerDecoder(vocab_size=V, max_len=max_len,
                           n_layers=N, d_model=D, n_heads=H, d_ff=D_ff)

# Dummy tokens
tokens = np.random.randint(0, V, size=(B, T))

# Forward pass
logits, attns = model.forward(tokens)

print("Input shape:", tokens.shape)
print("Logits shape:", logits.shape)       # Expect (2, 5, 1000)
print("Next-token probs shape:", softmax(logits[:, -1, :]).shape)  # Expect (2, 1000)
print("Attention weights shape:", attns[0].shape)  # Expect (2, 8, 5, 5)
