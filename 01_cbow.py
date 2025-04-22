import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore[import]
import matplotlib.pyplot as plt
import numpy as np

# Define the corpus
corpus = ['The cat sat on the mat',
          'The dog ran in the park',
          'The bird sang in the tree']

# Convert the corpus to a sequence of integers
from sklearn.preprocessing import LabelEncoder
from collections import Counter

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

print("Sequences of words in the corpus:", sequences)

# Parameters
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
embedding_size = 10  # Size of the word embeddings
window_size = 2  # Context window size

# Generate context-target pairs
contexts = []
targets = []

for sequence in sequences:
    for i in range(window_size, len(sequence) - window_size):
        context = sequence[i-window_size:i] + sequence[i+1:i+window_size+1]
        target = sequence[i]
        contexts.append(context)
        targets.append(target)

X = np.array(contexts)
y = np.array(targets)

# Create Dataset and DataLoader
class CBOWDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = torch.tensor(contexts, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]

dataset = CBOWDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define CBOW model in PyTorch
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        embedded = self.embeddings(inputs)  # Get embeddings for the context words
        # Take the average of the context word embeddings
        embedded_mean = embedded.mean(dim=1)
        out = self.linear(embedded_mean)  # Feed to linear layer to get word probabilities
        return out

# Initialize the model, loss function, and optimizer
model = CBOWModel(vocab_size=vocab_size, embedding_size=embedding_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for context, target in dataloader:
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader)}')