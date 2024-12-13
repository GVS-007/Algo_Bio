import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from argparse import Namespace
import os
import h5py

########################################
# Configuration
########################################

scratch_dir = "/scratch/vgunda8/algo_comp_bio_datasets"
data_file = 'BindingAffinityPrediction/TCREpitopePairs.csv'
epochs = 10
batch_size = 16
learning_rate = 1e-4
heads = 4
lin_size = 512
drop_rate = 0.25
num_runs = 5  # Number of runs to average metrics

# Make sure CUDA is available or else it uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert")
bert_model = BertModel.from_pretrained("wukevin/tcr-bert").to(device)

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

def generate_token_embeddings(sequences, file_name, max_length, batch_size=16):
    embedding_file = os.path.join(scratch_dir, file_name)
    total_samples = len(sequences)
    embedding_dim = 768

    if os.path.exists(embedding_file):
        print(f"Embeddings file {embedding_file} already exists. Using existing file.")
        return embedding_file
    else:
        with h5py.File(embedding_file, 'w') as f:
            embeddings = f.create_dataset('embeddings', shape=(total_samples, max_length, embedding_dim), dtype='float32')
            for idx in tqdm(range(0, total_samples, batch_size), desc=f"Processing {file_name}"):
                batch_seqs = sequences[idx:idx + batch_size]
                spaced_seqs = [' '.join(seq) for seq in batch_seqs]
                inputs = tokenizer(
                    spaced_seqs,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=max_length
                ).to(device)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.cpu().numpy()
                    embeddings[idx:idx + batch_size] = batch_embeddings
                del inputs, outputs, batch_embeddings
                torch.cuda.empty_cache()
        print(f"Saved embeddings to {embedding_file}")
        return embedding_file

########################################
# Dataset Class
########################################

class BindingDataset(Dataset):
    def __init__(self, tcr_embedding_file, epi_embedding_file, labels):
        self.labels = labels
        self.tcr_file = h5py.File(tcr_embedding_file, 'r')
        self.epi_file = h5py.File(epi_embedding_file, 'r')
        self.tcr_embeddings = self.tcr_file['embeddings']
        self.epi_embeddings = self.epi_file['embeddings']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tcr_embed = torch.tensor(self.tcr_embeddings[idx], dtype=torch.float32)
        epi_embed = torch.tensor(self.epi_embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tcr_embed, epi_embed, label

    def __del__(self):
        self.tcr_file.close()
        self.epi_file.close()

########################################
# Model Definition
########################################

class BiDirectionalCrossAttentionNet(nn.Module):
    def __init__(self, embedding_dim, args):
        super(BiDirectionalCrossAttentionNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.cross_attn_tcr = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads, batch_first=True)
        self.cross_attn_epi = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads, batch_first=True)

        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = args.lin_size

        self.net = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.size_hidden1_dense),
            nn.BatchNorm1d(self.size_hidden1_dense),
            nn.Dropout(args.drop_rate),
            nn.SiLU(),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.BatchNorm1d(self.size_hidden2_dense),
            nn.Dropout(args.drop_rate),
            nn.SiLU(),
            nn.Linear(self.size_hidden2_dense, 1),
            nn.Sigmoid()
        )

    def forward(self, tcr_embeds, epi_embeds):
        attn_output_tcr, _ = self.cross_attn_tcr(tcr_embeds, epi_embeds, epi_embeds)
        attn_output_epi, _ = self.cross_attn_epi(epi_embeds, tcr_embeds, tcr_embeds)

        attn_output_tcr_pooled = attn_output_tcr.mean(dim=1)
        attn_output_epi_pooled = attn_output_epi.mean(dim=1)

        combined_output = torch.cat((attn_output_tcr_pooled, attn_output_epi_pooled), dim=1)
        output = self.net(combined_output)
        return output

########################################
# Metrics Computation
########################################

def compute_metrics(all_labels, all_probs):
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

def run_experiment(data, split_method, max_length):
    # Only run tcr and epi splits as requested
    if split_method == "tcr":
        train_data, test_data, split_type = tcr_split(data)
    elif split_method == "epi":
        train_data, test_data, split_type = epi_split(data)
    elif split_method == "random":
        train_data, test_data, split_type = random_split(data)
    else:
        raise ValueError("Only 'tcr' and 'epi' splits are considered for multiple runs.")

    tcr_sequences_train = train_data['tcr'].tolist()
    epi_sequences_train = train_data['epi'].tolist()
    tcr_sequences_test = test_data['tcr'].tolist()
    epi_sequences_test = test_data['epi'].tolist()

    y_train = train_data['binding'].values
    y_test = test_data['binding'].values

    # Embeddings
    tcr_train_file = f"tcr_embeddings_train_{split_type}_len{max_length}.h5"
    epi_train_file = f"epi_embeddings_train_{split_type}_len{max_length}.h5"
    tcr_test_file = f"tcr_embeddings_test_{split_type}_len{max_length}.h5"
    epi_test_file = f"epi_embeddings_test_{split_type}_len{max_length}.h5"

    tcr_embeddings_train_file = generate_token_embeddings(tcr_sequences_train, tcr_train_file, max_length=max_length)
    epi_embeddings_train_file = generate_token_embeddings(epi_sequences_train, epi_train_file, max_length=max_length)
    tcr_embeddings_test_file = generate_token_embeddings(tcr_sequences_test, tcr_test_file, max_length=max_length)
    epi_embeddings_test_file = generate_token_embeddings(epi_sequences_test, epi_test_file, max_length=max_length)

    train_dataset = BindingDataset(tcr_embeddings_train_file, epi_embeddings_train_file, y_train)
    test_dataset = BindingDataset(tcr_embeddings_test_file, epi_embeddings_test_file, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    embedding_dim = 768
    args = Namespace(
        heads=heads,
        lin_size=lin_size,
        drop_rate=drop_rate,
        max_len_tcr=max_length,
        max_len_epi=max_length
    )

    model_nn = BiDirectionalCrossAttentionNet(embedding_dim, args).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_model_path = f"best_model_{split_type}_len{max_length}.pt"

    # Training
    for epoch in range(1, epochs+1):
        model_nn.train()
        running_loss = 0.0
        for tcr_embeds_batch, epi_embeds_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} ({split_type}, len={max_length})"):
            tcr_embeds_batch = tcr_embeds_batch.to(device)
            epi_embeds_batch = epi_embeds_batch.to(device)
            labels_batch = labels_batch.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model_nn(tcr_embeds_batch, epi_embeds_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * tcr_embeds_batch.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

        # Validation
        model_nn.eval()
        all_probs = []
        all_labels_col = []
        with torch.no_grad():
            for tcr_embeds_batch, epi_embeds_batch, labels_batch in test_loader:
                tcr_embeds_batch = tcr_embeds_batch.to(device)
                epi_embeds_batch = epi_embeds_batch.to(device)
                labels_batch = labels_batch.to(device).float().unsqueeze(1)
                outputs = model_nn(tcr_embeds_batch, epi_embeds_batch)
                probs = outputs.cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels_col.extend(labels_batch.cpu().numpy().flatten())

        accuracy, precision, recall, f1, auc = compute_metrics(all_labels_col, all_probs)
        print(f"Val Metrics -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model_nn.state_dict(), best_model_path)
            print("Saved Best Model")

    # Load Best Model and Final Evaluation
    model_nn.load_state_dict(torch.load(best_model_path))
    model_nn.eval()
    all_probs = []
    all_labels_col = []
    with torch.no_grad():
        for tcr_embeds_batch, epi_embeds_batch, labels_batch in test_loader:
            tcr_embeds_batch = tcr_embeds_batch.to(device)
            epi_embeds_batch = epi_embeds_batch.to(device)
            labels_batch = labels_batch.to(device).float().unsqueeze(1)
            outputs = model_nn(tcr_embeds_batch, epi_embeds_batch)
            probs = outputs.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels_col.extend(labels_batch.cpu().numpy().flatten())

    accuracy, precision, recall, f1, auc = compute_metrics(all_labels_col, all_probs)
    return precision, recall, f1, accuracy, auc, max_length

########################################
# Multiple Runs
########################################

def run_multiple_times(data, split_type, max_length, runs=5):
    precisions, recalls, f1s, accuracies, aucs = [], [], [], [], []
    for i in range(runs):
        print(f"Run {i+1}/{runs} for split={split_type}, max_length={max_length}")
        precision, recall, f1, accuracy, auc, ml = run_experiment(data, split_type, max_length)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
        aucs.append(auc)
    return {
        'Split': split_type,
        'Max_Length': ml,
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1-Score': np.mean(f1s),
        'Accuracy': np.mean(accuracies),
        'AUC': np.mean(aucs)
    }

splits_to_run = ["tcr", "epi", "random"]
max_lengths = [3]
results = []

for s in splits_to_run:
    for ml in max_lengths:
        results.append(run_multiple_times(data, s, ml, runs=num_runs))

results_df = pd.DataFrame(results)
print("\nFinal Results Table Over 5 Runs (TCR and EPI Splits):")
print(results_df)


# import pandas as pd
# import numpy as np
# from transformers import BertModel, BertTokenizer
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from argparse import Namespace
# import os
# import h5py
# import numpy as np
# ########################################
# # Configuration
# ########################################

# scratch_dir = "/scratch/vgunda8/algo_comp_bio_datasets"
# data_file = 'BindingAffinityPrediction/TCREpitopePairs.csv'
# epochs = 100
# batch_size = 16
# learning_rate = 1e-4
# heads = 4
# lin_size = 512
# drop_rate = 0.25

# # Make sure CUDA is available or else it uses CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)

# tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert")
# bert_model = BertModel.from_pretrained("wukevin/tcr-bert").to(device)

# ########################################
# # Data Loading
# ########################################

# data = pd.read_csv(data_file)
# data['binding'] = data['binding'].astype(int)

# ########################################
# # Helper Functions for Splitting
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

# def generate_token_embeddings(sequences, file_name, max_length, batch_size=16):
#     embedding_file = os.path.join(scratch_dir, file_name)
#     total_samples = len(sequences)
#     embedding_dim = 768  

#     if os.path.exists(embedding_file):
#         print(f"Embeddings file {embedding_file} already exists. Using existing file.")
#         return embedding_file
#     else:
#         with h5py.File(embedding_file, 'w') as f:
#             embeddings = f.create_dataset('embeddings', shape=(total_samples, max_length, embedding_dim), dtype='float32')
#             for idx in tqdm(range(0, total_samples, batch_size), desc=f"Processing {file_name}"):
#                 batch_seqs = sequences[idx:idx + batch_size]
#                 spaced_seqs = [' '.join(seq) for seq in batch_seqs]
#                 inputs = tokenizer(
#                     spaced_seqs,
#                     return_tensors='pt',
#                     padding='max_length',
#                     truncation=True,
#                     max_length=max_length
#                 ).to(device)
#                 with torch.no_grad():
#                     outputs = bert_model(**inputs)
#                     batch_embeddings = outputs.last_hidden_state.cpu().numpy()
#                     embeddings[idx:idx + batch_size] = batch_embeddings
#                 del inputs, outputs, batch_embeddings
#                 torch.cuda.empty_cache()
#         print(f"Saved embeddings to {embedding_file}")
#         return embedding_file

# ########################################
# # Dataset Class
# ########################################

# class BindingDataset(Dataset):
#     def __init__(self, tcr_embedding_file, epi_embedding_file, labels):
#         self.labels = labels
#         self.tcr_file = h5py.File(tcr_embedding_file, 'r')
#         self.epi_file = h5py.File(epi_embedding_file, 'r')
#         self.tcr_embeddings = self.tcr_file['embeddings']
#         self.epi_embeddings = self.epi_file['embeddings']

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         tcr_embed = torch.tensor(self.tcr_embeddings[idx], dtype=torch.float32)
#         epi_embed = torch.tensor(self.epi_embeddings[idx], dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return tcr_embed, epi_embed, label

#     def __del__(self):
#         self.tcr_file.close()
#         self.epi_file.close()

# ########################################
# # Model Definition
# ########################################

# class BiDirectionalCrossAttentionNet(nn.Module):
#     def __init__(self, embedding_dim, args):
#         super(BiDirectionalCrossAttentionNet, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.cross_attn_tcr = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads, batch_first=True)
#         self.cross_attn_epi = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads, batch_first=True)

#         self.size_hidden1_dense = 2 * args.lin_size
#         self.size_hidden2_dense = args.lin_size

#         self.net = nn.Sequential(
#             nn.Linear(2 * self.embedding_dim, self.size_hidden1_dense),
#             nn.BatchNorm1d(self.size_hidden1_dense),
#             nn.Dropout(args.drop_rate),
#             nn.SiLU(),
#             nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
#             nn.BatchNorm1d(self.size_hidden2_dense),
#             nn.Dropout(args.drop_rate),
#             nn.SiLU(),
#             nn.Linear(self.size_hidden2_dense, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, tcr_embeds, epi_embeds):
#         attn_output_tcr, _ = self.cross_attn_tcr(tcr_embeds, epi_embeds, epi_embeds)
#         attn_output_epi, _ = self.cross_attn_epi(epi_embeds, tcr_embeds, tcr_embeds)

#         attn_output_tcr_pooled = attn_output_tcr.mean(dim=1)
#         attn_output_epi_pooled = attn_output_epi.mean(dim=1)

#         combined_output = torch.cat((attn_output_tcr_pooled, attn_output_epi_pooled), dim=1)
#         output = self.net(combined_output)
#         return output

# ########################################
# # Training and Evaluation Function
# ########################################

# def run_experiment(data, split_method, max_length):
#     # Split data based on the method
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

#     # Unique file names for embeddings
#     tcr_train_file = f"tcr_embeddings_train_{split_type}_len{max_length}.h5"
#     epi_train_file = f"epi_embeddings_train_{split_type}_len{max_length}.h5"
#     tcr_test_file = f"tcr_embeddings_test_{split_type}_len{max_length}.h5"
#     epi_test_file = f"epi_embeddings_test_{split_type}_len{max_length}.h5"

#     # Generate Embeddings
#     tcr_embeddings_train_file = generate_token_embeddings(tcr_sequences_train, tcr_train_file, max_length=max_length)
#     epi_embeddings_train_file = generate_token_embeddings(epi_sequences_train, epi_train_file, max_length=max_length)
#     tcr_embeddings_test_file = generate_token_embeddings(tcr_sequences_test, tcr_test_file, max_length=max_length)
#     epi_embeddings_test_file = generate_token_embeddings(epi_sequences_test, epi_test_file, max_length=max_length)

#     # Create Datasets and Loaders
#     train_dataset = BindingDataset(tcr_embeddings_train_file, epi_embeddings_train_file, y_train)
#     test_dataset = BindingDataset(tcr_embeddings_test_file, epi_embeddings_test_file, y_test)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     # Model Setup
#     embedding_dim = 768  
#     args = Namespace(
#         heads=heads,
#         lin_size=lin_size,
#         drop_rate=drop_rate,
#         max_len_tcr=max_length,
#         max_len_epi=max_length
#     )

#     model_nn = BiDirectionalCrossAttentionNet(embedding_dim, args).to(device)
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)

#     best_accuracy = 0.0
#     best_model_path = f"best_model_{split_type}_len{max_length}.pt"

#     # Training
#     for epoch in range(1, epochs+1):
#         model_nn.train()
#         running_loss = 0.0
#         for tcr_embeds_batch, epi_embeds_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} ({split_type}, len={max_length})"):
#             tcr_embeds_batch = tcr_embeds_batch.to(device)
#             epi_embeds_batch = epi_embeds_batch.to(device)
#             labels_batch = labels_batch.to(device).float().unsqueeze(1)

#             optimizer.zero_grad()
#             outputs = model_nn(tcr_embeds_batch, epi_embeds_batch)
#             loss = criterion(outputs, labels_batch)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * tcr_embeds_batch.size(0)
#         epoch_loss = running_loss / len(train_dataset)
#         print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

#         # Validation
#         model_nn.eval()
#         correct = 0
#         total = 0
#         all_preds = []
#         all_labels_col = []
#         with torch.no_grad():
#             for tcr_embeds_batch, epi_embeds_batch, labels_batch in test_loader:
#                 tcr_embeds_batch = tcr_embeds_batch.to(device)
#                 epi_embeds_batch = epi_embeds_batch.to(device)
#                 labels_batch = labels_batch.to(device).float().unsqueeze(1)
#                 outputs = model_nn(tcr_embeds_batch, epi_embeds_batch)
#                 predicted = (outputs > 0.5).float()
#                 total += labels_batch.size(0)
#                 correct += (predicted == labels_batch).sum().item()
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels_col.extend(labels_batch.cpu().numpy())
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
#     all_labels_col = []
#     with torch.no_grad():
#         for tcr_embeds_batch, epi_embeds_batch, labels_batch in test_loader:
#             tcr_embeds_batch = tcr_embeds_batch.to(device)
#             epi_embeds_batch = epi_embeds_batch.to(device)
#             labels_batch = labels_batch.to(device).float().unsqueeze(1)
#             outputs = model_nn(tcr_embeds_batch, epi_embeds_batch)
#             predicted = (outputs > 0.5).float()
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels_col.extend(labels_batch.cpu().numpy())

#     # Classification Report
#     # report = classification_report(all_labels_col, all_preds, digits=4, output_dict=True)
# # After model evaluation and collecting all_preds, all_labels_col
#     report = classification_report(
#         all_labels_col, 
#         all_preds, 
#         digits=4, 
#         output_dict=True, 
#         labels=[0, 1],
#         zero_division=0  # Avoid undefined metrics errors
#     )

#     if '1' in report:
#         precision = report['1']['precision']
#         recall = report['1']['recall']
#         f1 = report['1']['f1-score']
#     else:
#         # If there's no class '1' at all, set metrics to 0.
#         precision = 0.0
#         recall = 0.0
#         f1 = 0.0

#     # Compute accuracy directly to avoid KeyError
#     all_preds_np = np.array(all_preds).flatten()
#     all_labels_np = np.array(all_labels_col).flatten()
#     accuracy = (all_preds_np == all_labels_np).mean()

#     return split_type, max_length, precision, recall, f1, accuracy

# ########################################
# # Run all experiments and store results
# ########################################

# split_types = ["random", "tcr", "epi"]
# max_lengths = [3, 5, 7]

# results = []

# for s in split_types:
#     for ml in max_lengths:
#         print(f"Running experiment for split={s}, max_length={ml}")
#         split_type, max_len_val, precision, recall, f1, accuracy = run_experiment(data, s, ml)
#         results.append({
#             'Split': split_type,
#             'Max_Length': max_len_val,
#             'Precision': precision,
#             'Recall': recall,
#             'F1-Score': f1,
#             'Accuracy': accuracy
#         })

# results_df = pd.DataFrame(results)
# print("\nFinal Results Table:")
# print(results_df)

