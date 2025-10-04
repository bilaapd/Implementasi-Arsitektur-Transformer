📘 Transformer From Scratch Using NumPy Implementation

Project ini adalah implementasi decoder-only Transformer (GPT-style) dari nol menggunakan NumPy.

Fokus utama adalah forward pass:

Input berupa token ID sederhana

Proses embedding + positional encoding

Multi-Head Attention dengan causal masking

Feed-Forward Network (FFN)

Residual connection + LayerNorm

Output berupa logits [batch, seq_len, vocab_size]

Distribusi probabilitas token berikutnya dengan softmax

⚙️ Fitur yang Diimplementasikan

✅ Token Embedding

✅ Positional Encoding

✅ Scaled Dot-Product Attention

✅ Multi-Head Attention 

✅ Feed-Forward Network 

✅ Residual Connection + LayerNorm 

✅ Causal Masking

✅ Output Layer ke vocab + softmax

🛠️ Cara Menjalankan
1. Clone Repository
git clone https://github.com/username/TransformerFromScratch.git
cd TransformerFromScratch

2. Install Dependencies

Pastikan sudah ada Python 3.9+. Lalu install NumPy:

pip install numpy

3. Jalankan Forward Pass Test
python test_forward.py

