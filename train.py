import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rnn_model import MultiTaskRNN
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import GPT2Tokenizer
import os

# Load a pre-trained tokenizer (you can use any model like GPT-2 or BERT)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token


device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE: ",device)

# Dataset
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, task):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        if task == "classify":
            self.labels = LabelEncoder().fit_transform([x["label"] for x in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.task == "classify":
            text = self.data[idx]["text"]
            tokens = self.tokenizer.encode(text , truncation=True, padding="max_length", max_length=512)
            if not tokens:
                tokens = [0]  # <PAD> token
            x = torch.tensor(tokens, dtype=torch.long)
            y = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            prompt = self.data[idx]["prompt"]
            continuation = self.data[idx]["continuation"]
            input_ids = self.tokenizer.encode(prompt )#, truncation=True, padding="max_length", max_length=32)#max_length = 512
            target_ids = self.tokenizer.encode(continuation)#, truncation=True, padding="max_length", max_length=32)# max_length= 512
            if not input_ids:
                input_ids = [0]
            if not target_ids:
                target_ids = [0]
            x = torch.tensor(input_ids, dtype=torch.long)
            y = torch.tensor(target_ids, dtype=torch.long)
        return x, y
        

# Collate function
def collate_fn(batch):
    batch = [(x, y) for x, y in batch if len(x) > 0]
    
    if len(batch) == 0:
        # fallback in case all samples are empty (rare)
        return None, None
    
    xs, ys = zip(*batch)
    # Replace empty sequences with a <PAD> token (index 0)
    xs = [x if x.numel() > 0 else torch.tensor([0]) for x in xs]
    ys = [y if y.numel() > 0 else torch.tensor([0]) for y in ys]
    xs = pad_sequence(xs, batch_first=True, padding_value=tokenizer.pad_token_id)
    #ys = pad_sequence(ys, batch_first=True, padding_value=0)
    ys = torch.tensor(ys) if isinstance(ys[0], torch.Tensor) and ys[0].dim() == 0 else pad_sequence(ys, batch_first=True, padding_value=tokenizer.pad_token_id)
    return xs.to(device), ys.to(device)

# Load JSON
def load_json(path):
    with open(path) as f:
        return json.load(f)

cls_data = load_json("data/classification_data.json")
gen_data = load_json("data/generation_data.json")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Optional: add padding token
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token, so using EOS token as padding


# Split classification data
cls_train_data, cls_test_data = train_test_split(cls_data, test_size=0.2, random_state=42)

# Split generation data
gen_train_data, gen_test_data = train_test_split(gen_data, test_size=0.2, random_state=42)

# Create datasets using the split data
cls_train_dataset = TextDataset(cls_train_data, tokenizer, "classify")
cls_test_dataset = TextDataset(cls_test_data, tokenizer, "classify")

gen_train_dataset = TextDataset(gen_train_data, tokenizer, "generate")
gen_test_dataset = TextDataset(gen_test_data, tokenizer, "generate")

# Create DataLoaders for both training and testing
cls_train_loader = DataLoader(cls_train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
cls_test_loader = DataLoader(cls_test_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

gen_train_loader = DataLoader(gen_train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
gen_test_loader = DataLoader(gen_test_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

#model = MultiTaskRNN(vocab_size=len(tokenizer.word2idx), embed_dim=64, hidden_dim=128, num_classes=3)
model = MultiTaskRNN(vocab_size=tokenizer.vocab_size, embed_dim=64, hidden_dim=128, num_classes=3)

model.to(device)

criterion_cls = nn.CrossEntropyLoss()
criterion_gen = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train():
    checkpoint_path = "checkpoint.pt"

    start_epoch = 0
    loss_history = {"classification": None, "generation": None}

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        loss_history = checkpoint.get("loss", loss_history)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch} with previous losses")
        print(f"  Classification loss: {loss_history['classification']:.4f}")
        print(f"  Generation loss:     {loss_history['generation']:.4f}")
    else:
        print("No checkpoint found. Starting fresh training.")

    model.to(device)
    
    for epoch in range(start_epoch,20):
        model.train()
        cls_running_loss = 0.0
        gen_running_loss = 0.0
        print(f"\nEpoch {epoch+1}")

        # Classification training
        for input, label in tqdm(cls_train_loader, desc="Classification",leave=False):
            if input is None or label is None:
                continue  # skip empty batch
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(input, task="classify")
            loss = criterion_cls(out, label)
            loss.backward()
            optimizer.step()
            cls_running_loss += loss.item()

                # Generation training
        for prompt, target in tqdm(gen_train_loader, desc="Generation"):
            prompt, target = prompt.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(prompt, task="generate")
            min_len = min(out.shape[1], target.shape[1]) - 1
            out = out[:, :min_len, :].reshape(-1, out.shape[-1])
            target = target[:, 1:min_len+1].reshape(-1)
            loss = criterion_gen(out, target)
            loss.backward()
            optimizer.step()
            gen_running_loss += loss.item()
        # Compute average losses
        avg_cls_loss = cls_running_loss / len(cls_train_loader)
        avg_gen_loss = gen_running_loss / len(gen_train_loader)

    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': {
        'classification': avg_cls_loss,
        'generation': avg_gen_loss
        }
    }
    torch.save(checkpoint, "checkpoint.pt")
        

def test(model, cls_loader, gen_loader, tokenizer, device):
    model.eval()
    total, correct = 0, 0
    print("\n🧪 Classification Test:")
    with torch.no_grad():
        for x, y in tqdm(cls_loader, desc="Classify", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x, task="classify")
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total if total > 0 else 0
    print(f"Classification Accuracy: {acc:.2f}%")

    print("\n🧪 Generation Test (Sample Outputs):")
    with torch.no_grad():
        for x, _ in gen_loader:
            x = x.to(device)
            out = model(x, task="generate")
            next_tokens = out.argmax(dim=-1)
            for i in range(min(3, len(x))):  # show up to 3 examples
                input_text = tokenizer.decode(x[i])
                generated = tokenizer.decode(next_tokens[i])
                print(f"> Prompt: {input_text}")
                print(f"> Generated: {generated}\n")
            break  # only one batch preview


if __name__ == "__main__":
    train()
    # Recreate the model and optimizer
# Load the checkpoint
    checkpoint = torch.load("checkpoint.pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.to(device)


    test(model, cls_test_loader, gen_test_loader, tokenizer, device)

