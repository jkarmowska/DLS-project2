import argparse
import json
from pathlib import Path

import torch

from restore_subtaskA import load_models, read_fasta, score_all, evaluate_mutation, choose_device


FASTA_PATH = Path("subtaskB.fa")
RESULTS_TSV = Path("subtaskB_restored_results.tsv")
METRICS_JSON = Path("subtaskB_restored_metrics.json")


def choose_best_mutation(models, sequence, current_scores, locked_positions):
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
            evaluated.append(evaluate_mutation(models, sequence, current_scores, candidate, model_name, candidate[0]))
        evaluated.sort(key=lambda item: item["avg_improvement"], reverse=True)

        for item in evaluated:
            if item["avg_improvement"] > 0.0 and item["lowered_models"] < 2:
                item["gradient_rank"] = rank + 1
                return item, rejected
            rejected.append(item)
    return None, rejected


def optimize_sequence(header, sequence, mutation_cap, voting_models, scoring_models):
    initial_scores = score_all(scoring_models, sequence)
    voting_initial_scores = score_all(voting_models, sequence)
    current_scores = dict(voting_initial_scores)
    current_sequence = sequence
    locked_positions = set()
    history = []
    rejected_count = 0

    for step in range(1, mutation_cap + 1):
        chosen, rejected = choose_best_mutation(voting_models, current_sequence, current_scores, locked_positions)
        rejected_count += len(rejected)
        if chosen is None:
            break

        current_sequence = chosen["sequence"]
        current_scores = chosen["scores"]
        locked_positions.add(chosen["position"])
        chosen.pop("sequence")
        chosen["step"] = step
        history.append(chosen)
        print(
            f"{header} step {step:03d}: {chosen['old_base']}{chosen['position'] + 1}{chosen['new_base']} "
            f"avg_delta={chosen['avg_improvement']:.5f}"
        )

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
        "mutation_cap": mutation_cap,
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
    parser = argparse.ArgumentParser(description="Optimize Subtask B with gradient-guided mutations.")
    parser.add_argument("--mutation-cap", type=int, default=100, help="Maximum introduced mutations.")
    parser.add_argument(
        "--exclude-jakubk",
        "--exclude-jk",
        action="store_true",
        dest="exclude_jakubk",
        help="Do not use jakubk model for mutation voting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = choose_device()
    scoring_models = load_models(device)
    voting_models = [
        model for model in scoring_models
        if not (args.exclude_jakubk and model.name.startswith("jakubk"))
    ]
    print("voting models:", ", ".join(model.name for model in voting_models))
    results = [
        optimize_sequence(header, sequence, args.mutation_cap, voting_models, scoring_models)
        for header, sequence in read_fasta(FASTA_PATH)
    ]
    save_results(results)
    print(f"saved {RESULTS_TSV}")
    print(f"saved {METRICS_JSON}")


if __name__ == "__main__":
    main()
