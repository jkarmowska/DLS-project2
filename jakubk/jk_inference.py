import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PREFIX_ADAPTER = "TCGGTTCACGCAATG"
SUFFIX_ADAPTER = "AGTTGATCCGGTCCT"
BASE_TO_CHANNEL = {"A": 0, "C": 1, "G": 2, "T": 3}


class DeepSTARRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, pool_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return self.dropout(x)


class DeepSTARRRegulatoryModel(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.conv_tower = nn.Sequential(
            DeepSTARRBlock(4, 256, kernel_size=7, dropout=0.10, pool_size=2),
            DeepSTARRBlock(256, 128, kernel_size=3, dropout=0.12, pool_size=2),
            DeepSTARRBlock(128, 128, kernel_size=3, dropout=0.15, pool_size=2),
            DeepSTARRBlock(128, 64, kernel_size=3, dropout=0.15, pool_size=1),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 4, sequence_length)
            flattened_size = int(np.prod(self.conv_tower(dummy).shape[1:]))

        self.shared_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
        )
        self.regression_head = nn.Linear(256, 1)
        self.classification_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_tower(x)
        x = self.shared_dense(x)
        return self.regression_head(x), self.classification_head(x)


def read_fasta(path):
    records = []
    header = None
    chunks = []

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks).upper()))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        records.append((header, "".join(chunks).upper()))

    return records


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint_path(checkpoint):
    path = Path(checkpoint)
    if path.exists():
        return path
    script_relative = Path(__file__).resolve().parent / checkpoint
    if script_relative.exists():
        return script_relative
    return path


def strip_known_flanks(sequence):
    sequence = sequence.upper()
    if sequence.startswith(PREFIX_ADAPTER) and sequence.endswith(SUFFIX_ADAPTER):
        return sequence[len(PREFIX_ADAPTER) : -len(SUFFIX_ADAPTER)]
    return sequence


def fit_sequence_length(sequence, expected_length):
    sequence = strip_known_flanks(sequence)
    if len(sequence) == expected_length:
        return sequence
    if len(sequence) > expected_length:
        start = (len(sequence) - expected_length) // 2
        return sequence[start : start + expected_length]
    left = (expected_length - len(sequence)) // 2
    right = expected_length - len(sequence) - left
    return ("N" * left) + sequence + ("N" * right)


def one_hot_sequence(sequence, expected_length, device):
    sequence = fit_sequence_length(sequence, expected_length)
    x = torch.zeros(1, 4, expected_length, dtype=torch.float32, device=device)

    for pos, base in enumerate(sequence):
        channel = BASE_TO_CHANNEL.get(base)
        if channel is not None:
            x[0, channel, pos] = 1.0

    return x, sequence


def reverse_complement_one_hot(x):
    return torch.flip(x[:, [3, 2, 1, 0], :], dims=[2])


def load_sequence(args):
    if args.sequence:
        return "cli_sequence", args.sequence.upper()

    records = read_fasta(args.fasta)
    if not records:
        raise ValueError(f"No FASTA records found in {args.fasta}")
    if args.record_index < 0 or args.record_index >= len(records):
        raise ValueError(f"--record-index must be between 0 and {len(records) - 1}")

    return records[args.record_index]


@torch.no_grad()
def predict(model, x, y_mean, y_std, use_rc_tta):
    model.eval()
    ratio_z, active_logit = model(x)

    if use_rc_tta:
        rc_ratio_z, rc_active_logit = model(reverse_complement_one_hot(x).contiguous())
        ratio_z = 0.5 * (ratio_z + rc_ratio_z)
        active_logit = 0.5 * (active_logit + rc_active_logit)

    ratio_value = (ratio_z.item() * y_std) + y_mean
    active_probability = active_logit.sigmoid().item()
    return ratio_value, active_probability


def main():
    parser = argparse.ArgumentParser(description="Run inference with the DeepSTARR rna_dna_ratio checkpoint.")
    parser.add_argument("--checkpoint", default="rna_dna_ratio_model.pt")
    parser.add_argument("--sequence", default=None, help="DNA sequence to score. If omitted, reads --fasta.")
    parser.add_argument("--fasta", default="subtaskA.fa")
    parser.add_argument("--record-index", type=int, default=0, help="0-based FASTA record index.")
    parser.add_argument("--rc-tta", action="store_true", help="Average prediction over forward and reverse-complement inputs.")
    args = parser.parse_args()

    device = choose_device()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_length = int(checkpoint["sequence_length"])
    y_mean = float(checkpoint["y_mean"])
    y_std = float(checkpoint["y_std"])

    header, raw_sequence = load_sequence(args)
    x, model_sequence = one_hot_sequence(raw_sequence, expected_length, device)

    model = DeepSTARRRegulatoryModel(expected_length).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ratio_value, active_probability = predict(model, x, y_mean, y_std, args.rc_tta)

    print(f"checkpoint: {checkpoint_path}")
    print(f"sequence_id: {header}")
    print(f"sequence_length: {len(model_sequence)}")
    print(f"device: {device}")
    print(f"predicted_rna_dna_ratio: {ratio_value:.6f}")
    print(f"predicted_is_active_probability: {active_probability:.6f}")


if __name__ == "__main__":
    main()
