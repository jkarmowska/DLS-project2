import argparse
import math
from pathlib import Path

import torch

from train_rna_dna_ratio import SimpleMultitaskPredictor, reverse_complement_one_hot


BASE_TO_CHANNEL = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


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


def one_hot_sequence(sequence, device):
    sequence = sequence.upper()
    x = torch.zeros(1, 4, len(sequence), dtype=torch.float32, device=device)

    for pos, base in enumerate(sequence):
        channel = BASE_TO_CHANNEL.get(base)
        if channel is None:
            raise ValueError(f"Unsupported base {base!r} at position {pos}")
        x[0, channel, pos] = 1.0

    return x


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sequence(args):
    if args.sequence:
        return "cli_sequence", args.sequence.upper()

    records = read_fasta(args.fasta)
    if not records:
        raise ValueError(f"No FASTA records found in {args.fasta}")
    if args.record_index < 0 or args.record_index >= len(records):
        raise ValueError(f"--record-index must be between 0 and {len(records) - 1}")

    return records[args.record_index]


def transformed_target_to_output(value, checkpoint_args):
    target_transform = checkpoint_args.get("target_transform", "none")
    if target_transform == "log2":
        log_eps = checkpoint_args.get("log_eps", 1e-6)
        return math.pow(2.0, value) - log_eps
    return value


@torch.no_grad()
def predict(model, x, target_mean, target_std, checkpoint_args, use_rc_tta):
    model.eval()
    ratio_pred, active_logit = model(x)

    if use_rc_tta:
        rc_ratio_pred, rc_active_logit = model(reverse_complement_one_hot(x).contiguous())
        ratio_pred = 0.5 * (ratio_pred + rc_ratio_pred)
        active_logit = 0.5 * (active_logit + rc_active_logit)

    ratio_value = (ratio_pred.item() * target_std) + target_mean
    ratio_value = transformed_target_to_output(ratio_value, checkpoint_args)
    active_prob = active_logit.sigmoid().item()
    return ratio_value, active_prob


def main():
    parser = argparse.ArgumentParser(description="Run inference with a saved rna_dna_ratio model checkpoint.")
    parser.add_argument("--checkpoint", default="rna_dna_ratio_model.pt")
    parser.add_argument("--sequence", default=None, help="DNA sequence to score. If omitted, reads --fasta.")
    parser.add_argument("--fasta", default="subtaskA.fa")
    parser.add_argument("--record-index", type=int, default=0, help="0-based FASTA record index.")
    parser.add_argument("--rc-tta", action="store_true", help="Average prediction over forward and reverse-complement inputs.")
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    dropout = checkpoint_args.get("dropout", 0.20)
    target_mean = checkpoint.get("target_mean", 0.0)
    target_std = checkpoint.get("target_std", 1.0)

    header, sequence = load_sequence(args)
    expected_length = checkpoint.get("sequence_length")
    if expected_length is not None and len(sequence) != expected_length:
        raise ValueError(f"Checkpoint expects length {expected_length}, but input has length {len(sequence)}")

    model = SimpleMultitaskPredictor(dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    x = one_hot_sequence(sequence, device)
    ratio_value, active_prob = predict(model, x, target_mean, target_std, checkpoint_args, args.rc_tta)

    print(f"checkpoint: {Path(args.checkpoint)}")
    print(f"sequence_id: {header}")
    print(f"sequence_length: {len(sequence)}")
    print(f"device: {device}")
    print(f"predicted_rna_dna_ratio: {ratio_value:.6f}")
    print(f"predicted_is_active_probability: {active_prob:.6f}")


if __name__ == "__main__":
    main()
