import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizerFast, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file, tokenizer_name='roberta-base', max_length=None):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Load and preprocess data
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            item = json.loads(line)
            data.append({
                'text': item['headline'],
                'label': item['is_sarcastic']
            })

        final_df = pd.DataFrame(data)
        final_df = self.preprocess(final_df)

        self.data = final_df

    def preprocess(self, df):
        # Remove numbers
        df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))

        # Lowercase
        df['text'] = df['text'].str.lower()

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data.iloc[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_text['input_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Modified RobertaLSTM class with dropout
class RobertaLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2, freeze_bert=True, dropout_rate=0.5):
        super(RobertaLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(self.roberta.config.hidden_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(roberta_output.last_hidden_state)
        pooled_output = torch.mean(lstm_output, 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


# Training function with early stopping mechanism
def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=3,
          early_stopping_patience=5):
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / len(train_loader.dataset)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_loss:.3f}, Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model_roberta.pt')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Evaluate function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, avg_accuracy
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data Loaders
    train_dataset = SarcasmDataset('Sarcasm_Headlines_Dataset.json')
    val_dataset = SarcasmDataset('Sarcasm_Headlines_Dataset_v2.json')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = RobertaLSTM().to(device)  # Change to RobertaRNN or RobertaMLP as needed

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=3,
          early_stopping_patience=5)
    print("Training complete!")

if __name__ == "__main__":
    main()
