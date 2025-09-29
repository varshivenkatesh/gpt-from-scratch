import torch
import torch.nn as nn
from torch.nn import functional as F
import string
import sentencepiece as spm

# hyperparameters - best for this encoding
batch_size = 48           # independent sequences processed in parallel - 32 if 384-1024
block_size = 512          # max context length for predictions, best for T4
max_iters = 8000
eval_interval = 500
learning_rate = 2e-4      # self-attention is not tolerant to high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # option to utilize GPU
eval_iters = 200
n_embd = 384              # number of embedding dim, increase if reducing batch size
n_head = 6                # if n_embd is 256, and 6 if n_embd is 384
n_layer = 6
dropout = 0.15
# ----------------------------------------

torch.manual_seed(1337) # for reproducibility while using randn

# wget https://raw.githubusercontent.com/varshivenkatesh/gpt-from-scratch/refs/heads/main/data/tiny_codes_data.txt
# read data to inspect
with open('tiny_codes_data.txt', 'r', encoding='utf-8') as f:
  text_tiny = f.read()
  
# unique characters that occur in the original text
# filter unwanted characters, keeping those relevant for Python coding

# Define allowed characters
allowed = set(
    string.ascii_letters +   # a-zA-Z
    string.digits +          # 0-9
    string.punctuation +     # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    " "                      # space
)

# Filter the dataset text to only allowed characters
text_tiny_fil = ''.join([c for c in text_tiny if c in allowed])

# Build vocab only from allowed chars
chars = sorted(list(set(text_tiny_fil)))
vocab_size = len(chars)

print("".join(chars))
print("Vocab size:", vocab_size)

# SentencePiece Tokenizer
# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='tiny_codes_data.txt',
    model_prefix='code_tokenizer',
    vocab_size=2000,  # adjust based on your needs
    character_coverage=1.0,  # important for code
    model_type='bpe',  # or 'unigram'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    # user_defined_symbols=['<pad>', '<unk>', '<s>', '</s>'],
    byte_fallback=True  # handles any character
)

# Load trained model
sp = spm.SentencePieceProcessor()
sp.load('code_tokenizer.model')

vocab_size = sp.get_piece_size()
print(f"Vocabulary size: {vocab_size}")

# Encode and decode functions
encode = lambda s: sp.encode_as_ids(s)
decode = lambda l: sp.decode_ids(l)

# train and val splits
data = torch.tensor(encode(text_tiny_fil), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading, batch dim
def get_batch(split):
  # generate small batch of data with inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

# avg out the losses over multiple batches
@torch.no_grad()          # no backpropagation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
  
# one head of self-attention
class Head(nn.Module):
  
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)     # (B,T,C)
    q = self.query(x)   # (B,T,C)
    
    # compute attention scores
    # scaled attention to normalize further
    weigh = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
    weigh = weigh.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
    weigh = F.softmax(weigh, dim=-1) # (B,T,T)
    weigh = self.dropout(weigh)
    
    # weighted aggregation of values
    v = self.value(x) # (B,T,C)
    out = weigh @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
    return out
  
# multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

# simple linear layer followed by a non-linearity
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd),  # per token level
      nn.ReLU(),
      nn.Linear(n_embd, n_embd),  # proj layer going back to residual pathway
      nn.Dropout(dropout),        # right before the connection back to residual pathway
    )
  
  def forward(self, x):
    return self.net(x)

# transformer block - communication followed by computation
class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    # n_embd: embedding dim, n_head: num of heads
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size) # communication
    self.ffwd = FeedForward(n_embd)                 # computation
    self.ln1 = nn.LayerNorm(n_embd)                 # size = 32
    self.ln2 = nn.LayerNorm(n_embd)
    
  def forward(self, x):           # residual connections
    x = x + self.sa(self.ln1(x))  # per token transformation
    x = x + self.ffwd(self.ln2(x))
    return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token form a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # tensor
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa_head = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dim self-attention
    # self.ffwd = FeedForward(n_embd) # (B,T,C)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer for tok_emb to logits

  def forward(self, idx, targets=None):
    B, T = idx.shape
    
    # idx and targets are both (B,T) tensor of integers
    # logits are prediction scores
    tok_emb = self.token_embedding_table(idx)                                 # (B,T,C) - batch by time by channel tensor
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))   # (T, C) - integers from 0 to T-1
    x = tok_emb + pos_emb       # (B,T,C)
    x = self.blocks(x)          # (B,T,C)
    logits = self.lm_head(x)    # (B,T,vocab_size)
    
    if targets is None:
        loss = None
    else: 
        # reshaping logits and targets as expected by PyTorch
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets) # comparing predictions and targets

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to last block_size tokens
      idx_cond = idx[:, -block_size:]
      
      # get predictions
      logits, loss = self(idx_cond)

      # focus only on the last time step
      logits = logits[:, -1, :] # (B, C)

      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)

      # sample from distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
  # sample a batch of data
  xb, yb = get_batch('train')
    
  # loss eval
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
