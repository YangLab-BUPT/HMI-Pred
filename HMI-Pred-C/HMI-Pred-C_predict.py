import os
import torch
import torch.nn.functional as F
import time
import csv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

class BinaryClassifier(nn.Module):
   
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return self.sigmoid(x)  

class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.flatten_size = (input_size // 8) * 64
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class EnsembleModel(nn.Module):
   
    def __init__(self, input_size):
        super(EnsembleModel, self).__init__()
        self.mlp = BinaryClassifier(input_size)
        self.cnn = CNNClassifier(input_size)
        self.lstm = LSTMClassifier(input_size)
        
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        
    def forward(self, x):
        mlp_out = self.mlp(x)
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(x)
        
        normalized_weights = F.softmax(self.weights, dim=0)
        
        ensemble_out = (0.4 * torch.sigmoid(mlp_out) + \
                      (0.3 * torch.sigmoid(cnn_out)) + \
                      (0.3 * torch.sigmoid(lstm_out)))

        return torch.logit(ensemble_out)


class CombinedTestDataset(Dataset):
    def __init__(self, positive_dir, negative_dir):
        self.positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.pt')]
        self.negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.pt')]
        self.all_files = self.positive_files + self.negative_files
        self.labels = [1] * len(self.positive_files) + [0] * len(self.negative_files)
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        filename = os.path.basename(filepath)
        feature_vector = torch.load(filepath)
        
        if isinstance(feature_vector, dict):
            feature_vector = [torch.tensor(v, dtype=torch.float32) for v in feature_vector['mean_representations'].values()]
        elif torch.is_tensor(feature_vector):
            feature_vector = [feature_vector]
        else:
            feature_vector = [torch.tensor(feature_vector, dtype=torch.float32)]
        
        return filename, feature_vector, self.labels[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testp_dir = 'esm_data/testp'
testn_dir = 'esm_data/testn'
max_length = 1280  
batch_size = 1000  

model = EnsembleModel(input_size=1280)
model.load_state_dict(torch.load('m_l.pth'))
model.eval()

dataset = CombinedTestDataset(testp_dir, testn_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

total_time = 0
all_predictions = []
all_filenames = []
all_labels = []

for batch_filenames, batch_feature_vectors, batch_labels in data_loader:
    batch_feature_vectors = [vec for vecs in batch_feature_vectors for vec in vecs]
    padded_feature_vectors = [F.pad(vec, (0, max_length - len(vec)), mode='constant', value=0) for vec in batch_feature_vectors]

    start_time = time.time()
    with torch.no_grad():
        predictions = model(torch.stack(padded_feature_vectors))
    prediction_time = time.time() - start_time
    
    total_time += round(prediction_time, 6)
    all_predictions.extend(predictions.tolist())
    all_filenames.extend(batch_filenames)
    all_labels.extend(batch_labels.tolist())

print(f"Total prediction time: {total_time:.4f} seconds")

csv_file = 'result.csv'
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Prediction', 'True_Label'])
    for filename, prediction, true_label in zip(all_filenames, all_predictions, all_labels):
        writer.writerow([filename, prediction[0], true_label])

print(f'All predictions saved to {csv_file}')