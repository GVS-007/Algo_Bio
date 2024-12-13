import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from argparse import Namespace
import h5py
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

########################################
# Configuration
########################################

scratch_dir = "/scratch/vgunda8/algo_comp_bio_datasets"
data_file = 'BindingAffinityPrediction/TCREpitopePairs.csv'
epochs = 10
batch_size = 32
learning_rate = 1e-4
num_runs = 5  # Number of runs to average metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert")
model = BertModel.from_pretrained("wukevin/tcr-bert").to(device)

########################################
# Data Loading
########################################

data = pd.read_csv(data_file)
data['binding'] = data['binding'].astype(int)

########################################
# Splitting Functions
########################################

def random_split(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    split_type = "random"
    return train_data, test_data, split_type

def tcr_split(data):
    unique_tcr = data['tcr'].unique()
    train_tcr, test_tcr = train_test_split(unique_tcr, test_size=0.2, random_state=42)
    train_data = data[data['tcr'].isin(train_tcr)]
    test_data = data[data['tcr'].isin(test_tcr)]
    split_type = "tcr"
    return train_data, test_data, split_type

def epi_split(data):
    unique_epi = data['epi'].unique()
    train_epi, test_epi = train_test_split(unique_epi, test_size=0.2, random_state=42)
    train_data = data[data['epi'].isin(train_epi)]
    test_data = data[data['epi'].isin(test_epi)]
    split_type = "epi"
    return train_data, test_data, split_type

########################################
# Embedding Generation
########################################

def generate_embeddings(sequences, embedding_file, batch_size=16):
    embedding_file = os.path.join(scratch_dir, embedding_file)
    try:
        embeddings = np.load(embedding_file)
        print(f"Loaded embeddings from {embedding_file}")
    except FileNotFoundError:
        embeddings = []
        max_length = 50  
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]
            spaced_seqs = [' '.join(seq) for seq in batch_seqs]
            inputs = tokenizer(
                spaced_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # CLS token representation
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
            del inputs, outputs, batch_embeddings
            torch.cuda.empty_cache()
        embeddings = np.vstack(embeddings)
        np.save(embedding_file, embeddings)
        print(f"Saved embeddings to {embedding_file}")
    return embeddings

########################################
# Dataset Class
########################################

class BindingDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

########################################
# Model Definition
########################################

class DeepNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

########################################
# Metrics Computation
########################################

def compute_metrics(all_labels, all_probs):
    # Convert to numpy
    all_labels_np = np.array(all_labels).flatten()
    all_probs_np = np.array(all_probs).flatten()
    all_preds_np = (all_probs_np > 0.5).astype(int)

    # Classification report
    report = classification_report(
        all_labels_np,
        all_preds_np,
        digits=4,
        output_dict=True,
        labels=[0, 1],
        zero_division=0
    )

    if '1' in report:
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    accuracy = (all_preds_np == all_labels_np).mean()

    # AUC
    if len(np.unique(all_labels_np)) == 2:
        auc = roc_auc_score(all_labels_np, all_probs_np)
    else:
        auc = 0.0

    return accuracy, precision, recall, f1, auc

########################################
# Single Run Experiment
########################################

def run_experiment(data, split_method):
    # Split data
    if split_method == "tcr":
        train_data, test_data, split_type = tcr_split(data)
    elif split_method == "epi":
        train_data, test_data, split_type = epi_split(data)    
    elif split_method == "random":
        train_data, test_data, split_type = random_split(data)
    else:
        raise ValueError("Only tcr and epi splits are considered for multiple runs.")

    tcr_sequences_train = train_data['tcr'].tolist()
    epi_sequences_train = train_data['epi'].tolist()
    tcr_sequences_test = test_data['tcr'].tolist()
    epi_sequences_test = test_data['epi'].tolist()

    y_train = train_data['binding'].values
    y_test = test_data['binding'].values

    print(f"Processing TCR embeddings for {split_type} training data...")
    base_tcr_embeddings_train = generate_embeddings(tcr_sequences_train, f'base_tcr_embeddings_train_{split_type}.npy')
    print(f"Processing epitope embeddings for {split_type} training data...")
    base_epi_embeddings_train = generate_embeddings(epi_sequences_train, f'base_epi_embeddings_train_{split_type}.npy')

    print(f"Processing TCR embeddings for {split_type} test data...")
    base_tcr_embeddings_test = generate_embeddings(tcr_sequences_test, f'base_tcr_embeddings_test_{split_type}.npy')
    print(f"Processing epitope embeddings for {split_type} test data...")
    base_epi_embeddings_test = generate_embeddings(epi_sequences_test, f'base_epi_embeddings_test_{split_type}.npy')

    X_train = np.concatenate([base_tcr_embeddings_train, base_epi_embeddings_train], axis=1)
    X_test = np.concatenate([base_tcr_embeddings_test, base_epi_embeddings_test], axis=1)

    train_dataset = BindingDataset(X_train, y_train)
    test_dataset = BindingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X_train.shape[1]
    num_classes = 2
    model_nn = DeepNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_model_path = f'base_best_model_{split_type}.pt'

    for epoch in range(1, epochs+1):
        model_nn.train()
        running_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} ({split_type})"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_nn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

        model_nn.eval()
        all_probs = []
        all_labels_col = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model_nn(batch_X)
                probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy() 
                all_probs.extend(probs)
                all_labels_col.extend(batch_y.cpu().numpy().flatten())

        accuracy, precision, recall, f1, auc = compute_metrics(all_labels_col, all_probs)
        print(f"Val Metrics -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model_nn.state_dict(), best_model_path)
            print("Saved Best Model")

    model_nn.load_state_dict(torch.load(best_model_path))
    model_nn.eval()
    all_probs = []
    all_labels_col = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model_nn(batch_X)
            probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
            all_probs.extend(probs)
            all_labels_col.extend(batch_y.cpu().numpy().flatten())

    accuracy, precision, recall, f1, auc = compute_metrics(all_labels_col, all_probs)
    return precision, recall, f1, accuracy, auc

########################################
# Multiple Runs
########################################

def run_multiple_times(data, split_type, runs=5):
    precisions, recalls, f1s, accuracies, aucs = [], [], [], [], []
    for i in range(runs):
        print(f"Run {i+1}/{runs} for split={split_type}")
        precision, recall, f1, accuracy, auc = run_experiment(data, split_type)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
        aucs.append(auc)

    return {
        'Split': split_type,
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1-Score': np.mean(f1s),
        'Accuracy': np.mean(accuracies),
        'AUC': np.mean(aucs)
    }

splits_to_run = ["tcr", "epi", "random"]
results = []

for s in splits_to_run:
    results.append(run_multiple_times(data, s, runs=5))

results_df = pd.DataFrame(results)
print("\nFinal Results Table Over 5 Runs (TCR and EPI Splits):")
print(results_df)


# import pandas as pd
# import numpy as np
# from transformers import BertModel, BertTokenizer
# import torch
# import os
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from argparse import Namespace
# import h5py
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# ########################################
# # Configuration
# ########################################

# scratch_dir = "/scratch/vgunda8/algo_comp_bio_datasets"
# data_file = 'BindingAffinityPrediction/TCREpitopePairs.csv'
# epochs = 100
# batch_size = 32
# learning_rate = 1e-4

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)

# tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert")
# model = BertModel.from_pretrained("wukevin/tcr-bert").to(device)

# ########################################
# # Data Loading
# ########################################

# data = pd.read_csv(data_file)
# data['binding'] = data['binding'].astype(int)

# ########################################
# # Splitting Functions
# ########################################

# def random_split(data):
#     train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#     split_type = "random"
#     return train_data, test_data, split_type

# def tcr_split(data):
#     unique_tcr = data['tcr'].unique()
#     train_tcr, test_tcr = train_test_split(unique_tcr, test_size=0.2, random_state=42)
#     train_data = data[data['tcr'].isin(train_tcr)]
#     test_data = data[data['tcr'].isin(test_tcr)]
#     split_type = "tcr"
#     return train_data, test_data, split_type

# def epi_split(data):
#     unique_epi = data['epi'].unique()
#     train_epi, test_epi = train_test_split(unique_epi, test_size=0.2, random_state=42)
#     train_data = data[data['epi'].isin(train_epi)]
#     test_data = data[data['epi'].isin(test_epi)]
#     split_type = "epi"
#     return train_data, test_data, split_type

# ########################################
# # Embedding Generation
# ########################################

# def generate_embeddings(sequences, embedding_file, batch_size=16):
#     embedding_file = os.path.join(scratch_dir, embedding_file)
#     try:
#         embeddings = np.load(embedding_file)
#         print(f"Loaded embeddings from {embedding_file}")
#     except FileNotFoundError:
#         embeddings = []
#         max_length = 50  
#         for i in tqdm(range(0, len(sequences), batch_size)):
#             batch_seqs = sequences[i:i + batch_size]
#             spaced_seqs = [' '.join(seq) for seq in batch_seqs]
#             inputs = tokenizer(
#                 spaced_seqs,
#                 return_tensors='pt',
#                 padding=True,
#                 truncation=True,
#                 max_length=max_length
#             ).to(device)
#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 # Taking the CLS token representation: [batch, seq_len, hidden_dim]
#                 # CLS is at index 0
#                 batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#                 embeddings.append(batch_embeddings)
#             del inputs, outputs, batch_embeddings
#             torch.cuda.empty_cache()
#         embeddings = np.vstack(embeddings)
#         np.save(embedding_file, embeddings)
#         print(f"Saved embeddings to {embedding_file}")
#     return embeddings

# ########################################
# # Dataset Class
# ########################################

# class BindingDataset(Dataset):
#     def __init__(self, features, labels):
#         self.X = torch.tensor(features, dtype=torch.float32)
#         self.y = torch.tensor(labels, dtype=torch.long)
#     def __len__(self):
#         return len(self.y)
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# ########################################
# # Model Definition
# ########################################

# class DeepNN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(DeepNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 1024)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, 512)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(512, 256)
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(0.5)
#         self.fc4 = nn.Linear(256, num_classes)
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         x = self.relu3(x)
#         x = self.dropout3(x)
#         x = self.fc4(x)
#         return x

# ########################################
# # Run Experiment Function
# ########################################

# def run_experiment(data, split_method):
#     # Split data
#     if split_method == "random":
#         train_data, test_data, split_type = random_split(data)
#     elif split_method == "tcr":
#         train_data, test_data, split_type = tcr_split(data)
#     elif split_method == "epi":
#         train_data, test_data, split_type = epi_split(data)
#     else:
#         raise ValueError("Invalid split method.")

#     tcr_sequences_train = train_data['tcr'].tolist()
#     epi_sequences_train = train_data['epi'].tolist()
#     tcr_sequences_test = test_data['tcr'].tolist()
#     epi_sequences_test = test_data['epi'].tolist()

#     y_train = train_data['binding'].values
#     y_test = test_data['binding'].values

#     print(f"Processing TCR embeddings for {split_type} training data...")
#     base_tcr_embeddings_train = generate_embeddings(tcr_sequences_train, f'base_tcr_embeddings_train_{split_type}.npy')
#     print(f"Processing epitope embeddings for {split_type} training data...")
#     base_epi_embeddings_train = generate_embeddings(epi_sequences_train, f'base_epi_embeddings_train_{split_type}.npy')

#     print(f"Processing TCR embeddings for {split_type} test data...")
#     base_tcr_embeddings_test = generate_embeddings(tcr_sequences_test, f'base_tcr_embeddings_test_{split_type}.npy')
#     print(f"Processing epitope embeddings for {split_type} test data...")
#     base_epi_embeddings_test = generate_embeddings(epi_sequences_test, f'base_epi_embeddings_test_{split_type}.npy')

#     # Concatenate embeddings
#     X_train = np.concatenate([base_tcr_embeddings_train, base_epi_embeddings_train], axis=1)
#     X_test = np.concatenate([base_tcr_embeddings_test, base_epi_embeddings_test], axis=1)

#     # Create datasets
#     train_dataset = BindingDataset(X_train, y_train)
#     test_dataset = BindingDataset(X_test, y_test)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     # Model setup
#     input_size = X_train.shape[1]  
#     num_classes = 2
#     model_nn = DeepNN(input_size, num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

#     best_accuracy = 0.0
#     best_model_path = f'base_best_model_{split_type}.pt'

#     # Training
#     for epoch in range(1, epochs+1):
#         model_nn.train()
#         running_loss = 0.0
#         for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} ({split_type})"):
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#             outputs = model_nn(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * batch_X.size(0)
#         epoch_loss = running_loss / len(train_dataset)
#         print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

#         # Validation
#         model_nn.eval()
#         correct = 0
#         total = 0
#         all_preds = []
#         all_labels = []
#         with torch.no_grad():
#             for batch_X, batch_y in test_loader:
#                 batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#                 outputs = model_nn(batch_X)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += batch_y.size(0)
#                 correct += (predicted == batch_y).sum().item()
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(batch_y.cpu().numpy())
#         accuracy = correct / total
#         print(f"Validation Accuracy: {accuracy:.4f}")
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             torch.save(model_nn.state_dict(), best_model_path)
#             print("Saved Best Model")

#     # Load Best Model and Evaluate
#     model_nn.load_state_dict(torch.load(best_model_path))
#     model_nn.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch_X, batch_y in test_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             outputs = model_nn(batch_X)
#             _, predicted = torch.max(outputs.data, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(batch_y.cpu().numpy())

#     # Classification Report
#     report = classification_report(all_labels, all_preds, digits=4, output_dict=True, zero_division=0)
#     precision = report['1']['precision'] if '1' in report else 0.0
#     recall = report['1']['recall'] if '1' in report else 0.0
#     f1 = report['1']['f1-score'] if '1' in report else 0.0
#     accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

#     return split_type, precision, recall, f1, accuracy

# ########################################
# # Run all experiments and store results
# ########################################

# split_methods = ["random", "tcr", "epi"]
# results = []

# for s in split_methods:
#     print(f"Running experiment for split={s}")
#     split_type, precision, recall, f1, accuracy = run_experiment(data, s)
#     results.append({
#         'Split': split_type,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1,
#         'Accuracy': accuracy
#     })

# results_df = pd.DataFrame(results)
# print("\nFinal Results Table:")
# print(results_df)
