import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    """Simple dataset of random sequences."""
    def __init__(self, num_samples=1000, seq_length=10, vocab_size=20):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.targets = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        logits = self.fc(out)
        return logits

def train(num_epochs=5):
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleRNN(dataset.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, dataset.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train()
