import argparse
import __main__
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt


BASES = "ACGT"
FASTA_PATH = Path("subtaskA.fa")
RESULTS_TSV = Path("subtaskA_restored_results.tsv")
METRICS_JSON = Path("subtaskA_restored_metrics.json")
MUTATION_BUDGETS = {
    "seq1_broken": 40,
    "seq_2_broken": 16,
    "seq_3_broken": 20,
}


class JuliaSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_conv = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.dropout_fc = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, 1)
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        x = torch.cat((avg_out, max_out), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return torch.sigmoid(self.classifier(x)).squeeze(), self.regressor(x).squeeze()


class DeepSTARR_MultiTask(nn.Module):
    def __init__(self, params, seq_len):
        super().__init__()
        self.params = params
        self.conv_layers = nn.ModuleList()
        in_channels = 4
        for i in range(1, params["n_conv_layer"] + 1):
            out_channels = params[f"num_filters{i}"]
            kernel_size = params[f"kernel_size{i}"]
            layer = [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=params["pad"]),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            if i > 1 and params["dropout_conv"] == "yes":
                layer.append(nn.Dropout(params["dropout_prob"]))
            self.conv_layers.append(nn.Sequential(*layer))
            in_channels = out_channels

        for _ in range(params["n_conv_layer"]):
            seq_len = seq_len // 2
        self.flatten_size = in_channels * seq_len
        self.dense_layers = nn.ModuleList()
        in_features = self.flatten_size
        for i in range(1, params["n_dense_layer"] + 1):
            out_features = params[f"dense_neurons{i}"]
            self.dense_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Dropout(params["dropout_prob"]),
                )
            )
            in_features = out_features
        self.out_is_active = nn.Linear(in_features, 1)
        self.out_rna_dna_ratio = nn.Linear(in_features, 1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        for layer in self.dense_layers:
            x = layer(x)
        return self.out_is_active(x), self.out_rna_dna_ratio(x)


class JakubKModel(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.AdaptiveMaxPool1d(1),
        )
        self.shared_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ratio_head = nn.Linear(32, 1)
        self.active_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x).squeeze(-1)
        x = self.shared_head(x)
        return self.ratio_head(x).squeeze(-1), self.active_head(x).squeeze(-1)


class ModelScorer:
    def __init__(self, name, model, base_to_channel, ratio_fn, device):
        self.name = name
        self.model = model.to(device).eval()
        self.base_to_channel = base_to_channel
        self.ratio_fn = ratio_fn
        self.device = device

    def encode(self, sequence):
        x = torch.zeros(1, 4, len(sequence), dtype=torch.float32, device=self.device)
        for pos, base in enumerate(sequence):
            x[0, self.base_to_channel[base], pos] = 1.0
        return x

    def ratio_tensor(self, x):
        return self.ratio_fn(self.model(x))

    def saliency(self, sequence, aggregate="abs_sum"):
        x = self.encode(sequence)
        x.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        out = self.ratio_tensor(x)
        out.backward()
        grad = x.grad[0].detach().cpu().numpy()  # shape (4, L)
        # The scale has the form (4, L), where 4 represents the modes (A, T, C, G or A, C, G, T)
        if aggregate == "abs_sum":
            # We compute the gradient only for the nucleotide that is physically present in the sequence (Gradient × Input)
            # x[0] has the shape (4, L) and contains 1.0 where the base is specified, and 0.0 elsewhere
            input_tensor = x[0].detach().cpu().numpy()
            sal = np.sum(grad * input_tensor, axis=0)  # Positive and negative values remain
            
        elif aggregate == "actual_base":
            sal = np.zeros(grad.shape[1], dtype=float)
            for pos, base in enumerate(sequence):
                ch = self.base_to_channel[base]
                sal[pos] = grad[ch, pos]
        else:
            input_tensor = x[0].detach().cpu().numpy()
            sal = np.sum(grad * input_tensor, axis=0)

        max_abs = np.max(np.abs(sal))
        if max_abs > 0:
            sal = sal / max_abs
        return sal
    
    def predict(self, sequence):
        with torch.no_grad():
            return float(self.ratio_tensor(self.encode(sequence)).item())

    def ranked_mutations(self, sequence, locked_positions):
        x = self.encode(sequence)
        x.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self.ratio_tensor(x).backward()
        grad = x.grad[0].detach().cpu()
        ranked = []
        for pos, old_base in enumerate(sequence):
            if pos in locked_positions:
                continue
            old_channel = self.base_to_channel[old_base]
            for new_base in BASES:
                if new_base != old_base:
                    new_channel = self.base_to_channel[new_base]
                    delta = grad[new_channel, pos] - grad[old_channel, pos]
                    ranked.append((float(delta), pos, old_base, new_base))
        ranked.sort(reverse=True)
        return ranked


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks).upper()))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    records.append((header, "".join(chunks).upper()))
    return records


def mutate(sequence, pos, new_base):
    return sequence[:pos] + new_base + sequence[pos + 1 :]


def transformed_jakubk_ratio(raw_ratio, checkpoint):
    args = checkpoint.get("args", {})
    value = raw_ratio * checkpoint.get("target_std", 1.0) + checkpoint.get("target_mean", 0.0)
    if args.get("target_transform", "none") == "log2":
        return torch.pow(torch.tensor(2.0, device=value.device), value) - args.get("log_eps", 1e-6)
    return value


def load_models(device):
    julias = JuliaSModel()
    julias.load_state_dict(torch.load("julias/best_model.pth", map_location=device, weights_only=False))

    __main__.DeepSTARR_MultiTask = DeepSTARR_MultiTask
    juliak = torch.load("juliak/best_model.pth", map_location=device, weights_only=False)

    jakubk_ckpt = torch.load("jakubk/rna_dna_ratio_model.pt", map_location=device, weights_only=False)
    jakubk = JakubKModel(jakubk_ckpt.get("args", {}).get("dropout", 0.2))
    jakubk.load_state_dict(jakubk_ckpt["model_state_dict"])

    return [
        ModelScorer("julias", julias, {"A": 0, "T": 1, "C": 2, "G": 3}, lambda out: out[1], device),
        ModelScorer("juliak", juliak, {"A": 0, "C": 1, "G": 2, "T": 3}, lambda out: out[1].squeeze(), device),
        ModelScorer(
            "jakubk",
            jakubk,
            {"A": 0, "C": 1, "G": 2, "T": 3},
            lambda out: transformed_jakubk_ratio(out[0], jakubk_ckpt).squeeze(),
            device,
        ),
    ]


def score_all(models, sequence):
    return {model.name: model.predict(sequence) for model in models}


def evaluate_mutation(models, sequence, current_scores, candidate, source_model, attribution_delta):
    _, pos, old_base, new_base = candidate
    new_sequence = mutate(sequence, pos, new_base)
    new_scores = score_all(models, new_sequence)
    improvements = {name: new_scores[name] - current_scores[name] for name in current_scores}
    lowered = sum(delta < 0.0 for delta in improvements.values())
    return {
        "source_model": source_model,
        "position": pos,
        "old_base": old_base,
        "new_base": new_base,
        "attribution_delta": attribution_delta,
        "avg_improvement": sum(improvements.values()) / len(improvements),
        "lowered_models": lowered,
        "scores": new_scores,
        "improvements": improvements,
        "sequence": new_sequence,
    }


def choose_mutation(models, sequence, current_scores, locked_positions):
    ranked_by_model = {
        model.name: model.ranked_mutations(sequence, locked_positions)
        for model in models
    }
    max_rank = min(len(ranked) for ranked in ranked_by_model.values())
    rejected = []
    for rank in range(max_rank):
        seen = set()
        candidates = []
        for model in models:
            candidate = ranked_by_model[model.name][rank]
            _, pos, _, new_base = candidate
            key = (pos, new_base)
            if key not in seen:
                candidates.append((model.name, candidate))
                seen.add(key)
        evaluated = []
        for model_name, candidate in candidates:
            delta = candidate[0]
            evaluated.append(evaluate_mutation(models, sequence, current_scores, candidate, model_name, delta))
        evaluated.sort(key=lambda item: item["avg_improvement"], reverse=True)
        for item in evaluated:
            if item["lowered_models"] < 2:
                item["gradient_rank"] = rank + 1
                return item, rejected
            rejected.append(item)


def restore_sequence(header, sequence, budget, voting_models, scoring_models):
    voting_initial_scores = score_all(voting_models, sequence)
    current_scores = dict(voting_initial_scores)
    current_sequence = sequence
    history = []
    rejected_count = 0
    locked_positions = set()
    for step in range(1, budget + 1):
        chosen, rejected = choose_mutation(voting_models, current_sequence, current_scores, locked_positions)
        rejected_count += len(rejected)
        current_sequence = chosen["sequence"]
        current_scores = chosen["scores"]
        locked_positions.add(chosen["position"])
        chosen.pop("sequence")
        chosen["step"] = step
        history.append(chosen)
        print(
            f"{header} step {step:02d}: {chosen['old_base']}{chosen['position'] + 1}{chosen['new_base']} "
            f"avg_delta={chosen['avg_improvement']:.5f}"
        )
    initial_scores = score_all(scoring_models, sequence)
    final_scores = score_all(scoring_models, current_sequence)
    improvements = {name: final_scores[name] - initial_scores[name] for name in initial_scores}
    voting_improvements = {
        name: current_scores[name] - voting_initial_scores[name]
        for name in voting_initial_scores
    }
    return {
        "id": header,
        "initial_sequence": sequence,
        "final_sequence": current_sequence,
        "budget": budget,
        "accepted_mutations": len(history),
        "rejected_candidates": rejected_count,
        "mutated_positions_1based": [pos + 1 for pos in sorted(locked_positions)],
        "voting_model_names": [model.name for model in voting_models],
        "scoring_model_names": [model.name for model in scoring_models],
        "initial_scores": initial_scores,
        "final_scores": final_scores,
        "improvements": improvements,
        "voting_initial_scores": voting_initial_scores,
        "voting_final_scores": current_scores,
        "voting_improvements": voting_improvements,
        "mean_initial_score": sum(initial_scores.values()) / len(initial_scores),
        "mean_final_score": sum(final_scores.values()) / len(final_scores),
        "mean_improvement": sum(improvements.values()) / len(improvements),
        "mean_voting_initial_score": sum(voting_initial_scores.values()) / len(voting_initial_scores),
        "mean_voting_final_score": sum(current_scores.values()) / len(current_scores),
        "mean_voting_improvement": sum(voting_improvements.values()) / len(voting_improvements),
        "history": history,
    }


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_results(results):
    model_names = list(results[0]["final_scores"].keys())
    with RESULTS_TSV.open("w") as handle:
        handle.write("id\tnew_sequence\tpredicted_rna_dna_ratio\t" + "\t".join(f"{name}_rna_dna_ratio" for name in model_names) + "\n")
        for result in results:
            mean_ratio = sum(result["final_scores"].values()) / len(model_names)
            ratios = "\t".join(f"{result['final_scores'][name]:.6f}" for name in model_names)
            handle.write(f"{result['id']}\t{result['final_sequence']}\t{mean_ratio:.6f}\t{ratios}\n")

    summary = {
        "results_tsv": str(RESULTS_TSV),
        "model_names": model_names,
        "voting_model_names": results[0]["voting_model_names"],
        "sequences": results,
        "mean_final_ensemble_score": sum(r["mean_final_score"] for r in results) / len(results),
        "mean_ensemble_improvement": sum(r["mean_improvement"] for r in results) / len(results),
        "mean_final_voting_score": sum(r["mean_voting_final_score"] for r in results) / len(results),
        "mean_voting_improvement": sum(r["mean_voting_improvement"] for r in results) / len(results),
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Restore Subtask A sequences with gradient-guided voting.")
    parser.add_argument("--exclude-jakubk", action="store_true", help="Do not use jakubk model for mutation voting.")
    parser.add_argument("--saliency-dir", default=None, help="Directory to save saliency maps (PNG and TSV) for each sequence and model.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = choose_device()
    scoring_models = load_models(device)
    voting_models = [
        model for model in scoring_models
        if not (args.exclude_jakubk and model.name == "jakubk")
    ]
    print("voting models:", ", ".join(model.name for model in voting_models))
    records = read_fasta(FASTA_PATH)
    results = [
        restore_sequence(header, sequence, MUTATION_BUDGETS[header], voting_models, scoring_models)
        for header, sequence in records
    ]

    # generate saliency maps if requested
    if args.saliency_dir is not None:
        os.makedirs(args.saliency_dir, exist_ok=True)
        for res in results:
            header = res.get("id")
            initial_seq = res.get("initial_sequence")
            final_seq = res.get("final_sequence")
            safe_header = header.replace("/", "_")
            for model in scoring_models:
                for label, seq in (("initial", initial_seq), ("final", final_seq)):
                    sal = model.saliency(seq)
                    tsv_path = os.path.join(args.saliency_dir, f"{safe_header}_{model.name}_{label}_saliency.tsv")
                    png_path = os.path.join(args.saliency_dir, f"{safe_header}_{model.name}_{label}_saliency.png")
                    # save TSV
                    with open(tsv_path, "w") as fh:
                        fh.write("position\tsaliency\n")
                        for i, v in enumerate(sal, start=1):
                            fh.write(f"{i}\t{v:.6f}\n")
                    # save PNG heatmap
                    plt.figure(figsize=(max(6, len(sal)/50), 2))
                    plt.imshow(sal[np.newaxis, :], aspect="auto", cmap="seismic", vmin=-1, vmax=1)
                    plt.colorbar(label="normalized saliency")
                    plt.yticks([])
                    plt.xlabel("position")
                    plt.title(f"{safe_header} - {model.name} ({label})")
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=150)
                    plt.close()
    save_results(results)
    print(f"saved {RESULTS_TSV}")
    print(f"saved {METRICS_JSON}")


if __name__ == "__main__":
    main()
