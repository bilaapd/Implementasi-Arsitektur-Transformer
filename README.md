# 📘 Transformer From Scratch Using Numpy

## 📌 Deskripsi
Project ini adalah implementasi **decoder-only Transformer (GPT-style)** dari nol menggunakan **NumPy**.  

Fokus utama adalah **forward pass**:  
- Input berupa token ID sederhana  
- Proses embedding + positional encoding  
- Multi-Head Attention dengan causal masking  
- Feed-Forward Network  
- Residual connection + LayerNorm  
- Output berupa logits `[batch, seq_len, vocab_size]`  
- Distribusi probabilitas token berikutnya dengan softmax  

---

## ⚙️ Fitur yang Diimplementasikan
- ✅ **Token Embedding**  
- ✅ **Positional Encoding** (sinusoidal)  
- ✅ **Scaled Dot-Product Attention**  
- ✅ **Multi-Head Attention**  
- ✅ **Feed-Forward Network**  
- ✅ **Residual Connection + LayerNorm (Pre-Norm)**  
- ✅ **Causal Masking**  
- ✅ **Output Layer** ke vocab + softmax  

---

## 🛠️ Cara Menjalankan

### 1. Clone Repository
```bash
git clone [https://github.com/bilaapd//Implementasi-Arsitektur-Transformer]
cd /Implementasi-Arsitektur-Transformer
```
### 2. Install Dependencies
Pastikan sudah ada Python 3.9+.
Install NumPy dengan:
```bash
pip install numpy
```
### 3. Jalankan Forward Pass Test
Gunakan perintah berikut untuk menjalankan tes sederhana:
```bash
python testforward.py
```
### 4. Output yang Diharapkan
Jika berhasil, output akan seperti berikut:
```bash
Input shape: (x, y)
Logits shape: (x, y, z)
Next-token probs shape: (x, y)
Attention weights shape: (w, x, y, z)
```
