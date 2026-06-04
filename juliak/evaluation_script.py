"""
Usage: python evaluation_script.py <path_to_model> <path_to_test_data>

Loads a trained model and generates predictions for a test dataset.
Output format: id\tpredicted_is_active\tpredicted_rna_dna_ratio
"""

import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from train import DeepSTARR_MultiTask



# ==========================================
# Model Architecture Definition
# Required here so torch.load() knows how to reconstruct the object.
# ==========================================
class DeepSTARR_MultiTask(nn.Module):
    def __init__(self, params, seq_len):
        super(DeepSTARR_MultiTask, self).__init__()

        self.params = params
        self.conv_layers = nn.ModuleList()
        in_channels = 4

        for i in range(1, params['n_conv_layer'] + 1):
            out_channels = params[f'num_filters{i}']
            kernel_size = params[f'kernel_size{i}']

            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=params['pad'])
            bn = nn.BatchNorm1d(out_channels)
            relu = nn.ReLU()
            pool = nn.MaxPool1d(2)

            layer_components = [conv, bn, relu, pool]
            if i > 1 and params['dropout_conv'] == 'yes':
                layer_components.append(nn.Dropout(params['dropout_prob']))

            self.conv_layers.append(nn.Sequential(*layer_components))
            in_channels = out_channels

        for _ in range(params['n_conv_layer']):
            seq_len = seq_len // 2
        self.flatten_size = in_channels * seq_len

        self.dense_layers = nn.ModuleList()
        in_features = self.flatten_size
        for i in range(1, params['n_dense_layer'] + 1):
            out_features = params[f'dense_neurons{i}']
            dense = nn.Linear(in_features, out_features)
            bn = nn.BatchNorm1d(out_features)
            relu = nn.ReLU()
            drop = nn.Dropout(params['dropout_prob'])

            self.dense_layers.append(nn.Sequential(dense, bn, relu, drop))
            in_features = out_features

        self.out_is_active = nn.Linear(in_features, 1)
        self.out_rna_dna_ratio = nn.Linear(in_features, 1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = torch.flatten(x, 1)

        for layer in self.dense_layers:
            x = layer(x)

        logits_is_active = self.out_is_active(x)
        pred_rna_dna_ratio = self.out_rna_dna_ratio(x)

        return logits_is_active, pred_rna_dna_ratio

# ============== DNA Sequence Encoding ==============

def one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    encoded = [mapping.get(base.upper(), [0,0,0,0]) for base in sequence]
    # Transpose to get (Channels, Length) format required by PyTorch Conv1d
    return torch.tensor(encoded, dtype=torch.float32).T


# ============== Main Evaluation ==============

def evaluate(model_path, test_data_path, device='cpu'):
    # Load the trained model
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Load the test dataset
    df = pd.read_csv(test_data_path, sep='\t')

    print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")

    with torch.no_grad():
        for index, row in df.iterrows():
            seq_id = row['id'] 
            sequence = row['sequence']

            # Prepare input tensor: shape (1, 4, Sequence_Length)
            input_tensor = one_hot_encode(sequence).unsqueeze(0).to(device)

            # Generate raw predictions
            logits_active, pred_ratio = model(input_tensor)

            # Process Binary Classification: Convert logit to probability, then threshold at 0.5
            prob_active = torch.sigmoid(logits_active).item()
            predicted_is_active = 1 if prob_active >= 0.5 else 0

            # Process Continuous Regression: Extract raw value
            predicted_rna_dna_ratio = pred_ratio.item()

            print(f"{seq_id}\t{predicted_is_active}\t{predicted_rna_dna_ratio:.4f}")
    

    
path_to_model = "best_model.pth"
path_to_test_data = "test.tsv"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluation_script.py <path_to_model> <path_to_test_data>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    evaluate(model_path, test_data_path, device=device)
