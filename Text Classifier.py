from torchtext.datasets import IMDB
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

train_data = list(IMDB(split='train'))
test_data = list(IMDB(split='test'))

tokenizer = get_tokenizer('basic_english')

def yeild_tokens(data):
  for _, text in data:
    yield tokenizer(text)

vocab = build_vocab_from_iterator(yeild_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

class TextDataset(Dataset):
  def __init__(self, data, vocab, tokenizer):
    self.data = data
    self.vocab = vocab
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    label, text = self.data[idx]
    tokens = self.tokenizer(text)
    token_ids = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)
    return token_ids, torch.tensor(1 if label == 'pos' else 0, dtype=torch.float)

def collate_fn(batch):
  texts, labels = zip(*batch)
  texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
  labels = torch.tensor(labels, dtype=torch.float)
  return texts, labels

train_dataset = TextDataset(train_data, vocab, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

class TextClassifier(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim):
    super(TextClassifier, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 1)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.embedding(x)
    _, (h_n, _) = self.lstm(x)
    out = self.fc(h_n[-1])
    return self.sigmoid(out)

vocab_size = len(vocab)
embed_dim = 16
hidden_dim = 32
model = TextClassifier(vocab_size, embed_dim, hidden_dim)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
  for texts, labels in train_loader:
    outputs = model(texts).squeeze()
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def predict(text):
  model.eval()
  tokens = tokenizer(text)
  token_ids = torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(0)
  with torch.no_grad():
    output = model(token_ids).item()
  return output

print(predict('This movie is great!'))
print(predict('This movie is terrible!'))