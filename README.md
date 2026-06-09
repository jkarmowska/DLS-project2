# DLS-project2

This repo contains the final DNA sequence optimization scripts for the restoration and super-enhancer tasks.

## Approach

Both subtasks use gradient attribution with an ensemble of three trained models (`julias`, `juliak`, `jakubk`) that predict `rna_dna_ratio`. At each step, gradients of the predicted ratio with respect to the one-hot DNA input are used to rank possible single-base substitutions.

The procedure for :

1. Compute gradient attribution for each model on the current sequence.
2. For each model, take its most promising mutation from the gradient-ranked list.
3. Score these candidate mutations with all voting models and choose the mutation with the largest average predicted improvement across the voting models.
5. If two models disagree, the mutation is rejected.
6. If the best candidate is rejected, try the next-best gradient-ranked candidate set.
7. Apply the accepted mutation and repeat until the mutation budget/cap is reached.

For Subtask A, the script restores the three broken enhancer sequences under the given mutation budgets. 

For Subtask B, the script starts from the inactive sequence and repeatedly applies gradient-guided mutations up to the configured mutation cap, defaulting to 100.

## Outputs

Restored sequences and model scores are written to:

- `subtaskA_restored_results.tsv`
- `subtaskA_restored_metrics.json`
- `subtaskB_restored_results.tsv`
- `subtaskB_restored_metrics.json`

## Visualizations

Additional optimization visualizations and saliency-map outputs are stored in `visualisations/`, `plots_output/`, and `saliency_outputs/`.
