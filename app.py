import streamlit as st
import torch
import json
import math
import torch.nn as nn

# ---------------------------
# Load vocabulary from JSON file
# ---------------------------
with open('vocab (3).json', 'r') as f:
    vocab = json.load(f)

# Create idx_to_word mapping from loaded vocabulary
idx_to_word = {idx: word for word, idx in vocab.items()}

# ---------------------------
# Tokenization function
# ---------------------------
def tokenize(text, vocab, max_length):
    tokens = text.lower().split()
    token_ids = [vocab.get(token, vocab['[UNK]']) for token in tokens]
    token_ids = token_ids[:max_length] + [vocab['[PAD]']] * (max_length - len(token_ids))
    return token_ids

# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ---------------------------
# Transformer Model Definition
# ---------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1, max_len=5000):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, 
                                          num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def encode(self, src):
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        memory = self.transformer.encoder(src_emb)
        return memory

    def decode(self, tgt, memory):
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt.device)
        output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def forward(self, src, tgt):
        memory = self.encode(src)
        return self.decode(tgt, memory)

# ---------------------------
# Load the trained model weights
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSeq2Seq(len(vocab), len(vocab))
model.load_state_dict(torch.load('psedocode-to-code.pth', map_location=device))
model.to(device)
model.eval()  # Ensure model is in evaluation mode

# ---------------------------
# Inference function: Process input pseudo-code line by line
# ---------------------------
def inference_line_by_line(model, input_pseudo_code, vocab, idx_to_word, device, max_length):
    model.eval()
    generated_lines = []

    for line in input_pseudo_code.strip().split("\n"):
        line = line.strip()  # Remove leading/trailing whitespace
        if not line:  # Ignore empty lines
            continue

        input_tokens = tokenize(line, vocab, max_length)
        input_tensor = torch.tensor([vocab['[pre]']] + input_tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            memory = model.encode(input_tensor)

        tgt_tokens = [vocab['[pre]']]
        output_sentence = []

        for _ in range(max_length):
            tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model.decode(tgt_tensor, memory)

            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            tgt_tokens.append(next_token)

            word = idx_to_word.get(next_token, '[UNK]')

            if word == "[endl]":  
                break

            if word == "[end]":  
                output_sentence.append("\n")
            else:
                output_sentence.append(word)

        generated_lines.append(" ".join(output_sentence))

    return "\n".join(generated_lines)


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Pseudo-code to Code Transformer")

# Define max_length for tokenization and generation
max_length = 62  

# Text area for pseudo-code input
pseudo_code = st.text_area("Enter your pseudo-code here:", 
"""
""", height=200)

# Button to trigger inference
if st.button("Generate Code"):
    output_result = inference_line_by_line(model, pseudo_code, vocab, idx_to_word, device, max_length)
    st.subheader("Generated Output:")
    st.text_area("", output_result, height=200)
