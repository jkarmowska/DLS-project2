import random
import numpy as np
import pandas as pd
import torch 

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


# DNA Sequence Encoding
def dna_one_hot(seq):
    """Converts a DNA sequence into a one-hot encoded matrix (4, L)."""
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    matrix = np.zeros((4, len(seq)))
    for i, base in enumerate(seq):
        if base in mapping:
            matrix[mapping[base], i] = 1
    return matrix.astype(np.float32)

def reverse_complement(seq):
    """Returns the reverse complement of a DNA string."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    # Reverse the string and swap bases
    return "".join(complement.get(base, base) for base in reversed(seq))

class DNADataset(Dataset):
    """Custom Dataset for DNA sequences with optional data augmentation."""
    def __init__(self, dataframe, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row['sequence']
        
        # Data Augmentation: 50% chance to use the reverse complement
        if self.augment and np.random.random() > 0.5:
            sequence = reverse_complement(sequence)
            
        seq_encoded = dna_one_hot(sequence)
        
        # Target labels for classification and regression
        is_active = torch.tensor(row['is_active'], dtype=torch.float32)
        ratio = torch.tensor(row['rna_dna_ratio'], dtype=torch.float32)
        
        return torch.tensor(seq_encoded), is_active, ratio

class DNAModel(nn.Module):
    """Hybrid CNN model for classification and regression on DNA sequences."""
    def __init__(self):
        super(DNAModel, self).__init__()
        
        # Convolutional layers for feature extraction from DNA motifs
        self.conv1 = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Downsampling and global feature aggregation
        self.pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Regularization to prevent overfitting
        self.dropout_conv = nn.Dropout(0.3)
        
        # Fully connected layer (receives concatenated avg and max pooling outputs)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.dropout_fc = nn.Dropout(0.3)
        
        # Dual output heads
        self.classifier = nn.Linear(64, 1) # Binary classification (active/inactive)
        self.regressor = nn.Linear(64, 1)  # Continuous regression (strength)

    def forward(self, x):
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.dropout_conv(x)
        
        # Multi-strategy pooling: combine average and maximum features
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        x = torch.cat((avg_out, max_out), dim=1) 
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        # Return outputs for both tasks
        return torch.sigmoid(self.classifier(x)).squeeze(), self.regressor(x).squeeze()


if __name__ == "__main__":
    # --- Data Preparation ---
    df = pd.read_csv('dataset.tsv', sep='\t')
    
    # Split data: 70% Train, 30% Temporary (Validation + Test)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    
    # Split Temporary: 50% Validation (15% total), 50% Test (15% total)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    batch_size = 64

    # Initialize datasets (Augmentation only for training)
    train_dataset = DNADataset(train_df, augment=True)  
    val_dataset = DNADataset(val_df, augment=False)
    test_dataset = DNADataset(test_df, augment=False)

    # Prepare DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Setup ---
    model = DNAModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    
    # Dynamic learning rate reduction on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    criterion_cls = nn.BCELoss() # Binary Cross Entropy for classification
    criterion_reg = nn.MSELoss()  # Mean Squared Error for regression

    # Early stopping configuration
    patience = 10
    best_val_loss = float('inf')
    counter = 0 
    path_to_saved_model = 'best_model.pth'

    # --- Training Loop ---
    for epoch in range(100):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_correct = 0
        train_total = 0

        for sequences, labels_cls, labels_reg in train_loader:
            optimizer.zero_grad()
            pred_cls, pred_reg = model(sequences)
            
            # Combine losses from both tasks
            loss_cls = criterion_cls(pred_cls, labels_cls)
            loss_reg = criterion_reg(pred_reg, labels_reg)
            loss = loss_cls + loss_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += loss_reg.item()
            
            # Track training accuracy
            preds_binary = (pred_cls > 0.5).float()
            train_correct += (preds_binary == labels_cls).sum().item()
            train_total += labels_cls.size(0)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_correct = 0 
        val_total = 0

        with torch.no_grad():
            for v_seq, v_cls, v_reg in val_loader:
                p_cls, p_reg = model(v_seq)
                v_loss_cls = criterion_cls(p_cls, v_cls)
                v_loss_reg = criterion_reg(p_reg, v_reg)
                
                val_loss += (v_loss_cls + v_loss_reg).item()
                val_mse += v_loss_reg.item()
                
                preds_binary = (p_cls > 0.5).float()
                val_correct += (preds_binary == v_cls).sum().item()
                val_total += v_cls.size(0)

        # Metrics calculation
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mse = train_mse / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/100")
        print(f"    TRAIN | Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, MSE: {avg_train_mse:.4f}")
        print(f"    VAL   | Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, MSE: {avg_val_mse:.4f}")

        # Check for improvement (Early Stopping logic)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0 
            torch.save(model.state_dict(), path_to_saved_model)
            print(f"Saved new best model")
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # --- FINAL TEST EVALUATION ---
    print("\n" + "="*30)
    print("RUNNING FINAL TEST EVALUATION")
    print("="*30)
    
    # Load the best weights from the training
    model.load_state_dict(torch.load(path_to_saved_model))
    model.eval()
    
    t_correct = 0
    t_total = 0
    t_mse = 0.0

    with torch.no_grad():
        for t_seq, t_cls, t_reg in test_loader:
            tp_cls, tp_reg = model(t_seq)
            t_mse += criterion_reg(tp_reg, t_reg).item()
            t_preds_binary = (tp_cls > 0.5).float()
            t_correct += (t_preds_binary == t_cls).sum().item()
            t_total += t_cls.size(0)

    print(f"FINAL TEST RESULTS:")
    print(f"Test Accuracy: {t_correct / t_total:.4f}")
    print(f"Test MSE:      {t_mse / len(test_loader):.4f}")