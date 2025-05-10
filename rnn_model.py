import torch
import torch.nn as nn

class MultiTaskRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(MultiTaskRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        # Task 1: Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Task 2: Next word generation head
        self.generator = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, task="classify"):
        x = self.embedding(x)
        out, _ = self.rnn(x)

        if task == "classify":
            return self.classifier(out[:, -1, :])  # Use last hidden state
        elif task == "generate":
            return self.generator(out)  # Return full sequence predictions
        else:
            raise ValueError("Invalid task type")
